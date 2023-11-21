import numpy as np
import parking_model as pm
from functools import reduce

class State:
    def __init__(self, state_vector: list):
        self.x = state_vector[0]
        self.y = state_vector[1]
        self.car_angle = state_vector[2]

class HiperParameters(object):
   def __init__(self, grid_width, grid_height, grid_offsets: [(float, float)], num_of_car_angle_values, num_of_wheel_angle_values, velocities):
       self.grid_width = grid_width
       self.grid_height = grid_height
       self.grid_offsets = grid_offsets
       self.num_of_car_angle_values = num_of_car_angle_values
       self.grid_num_of_wheel_angle_values = num_of_wheel_angle_values
       self.velocities = velocities

class CoverageEncoder(object):
    def __init__(self, global_variables: pm.GlobalVar, hiper_parameters: HiperParameters):
        self.global_variables = global_variables
        self.hiper_parameters = hiper_parameters
        self.field_width = global_variables.street_length / hiper_parameters.grid_width
        self.field_height = global_variables.street_width / hiper_parameters.grid_height
        self.delta_car_angle = 2.0 * np.pi / hiper_parameters.num_of_car_angle_values
        self.car_angles = np.arange(-np.pi, np.pi, self.delta_car_angle)
        self.delta_wheel_angle = 2.0 * global_variables.wheel_turn_angle_max / hiper_parameters.grid_num_of_wheel_angle_values
        self.wheel_angles = np.arange(-global_variables.wheel_turn_angle_max, global_variables.wheel_turn_angle_max, self.delta_wheel_angle)
        print('car_angles =', self.car_angles)
        print('wheel_angles =', self.wheel_angles)

    def count_weights(self):
        return reduce(lambda x, y: x * y, self.get_weights_shape(), 1)

    def get_weights_shape(self) -> [int]:
        return [
            len(self.hiper_parameters.grid_offsets),
            self.hiper_parameters.grid_width,
            self.hiper_parameters.grid_height,
            self.hiper_parameters.num_of_car_angle_values,
            len(self.get_actions())
        ]

    def get_actions(self) -> [(float, float)]:
        wheel_turn_actions = np.tile(self.wheel_angles, len(self.hiper_parameters.velocities))
        velocity_actions = np.repeat(self.hiper_parameters.velocities, len(self.wheel_angles))
        return np.column_stack((wheel_turn_actions, velocity_actions))

    def get_state_projections(self, state: State):
        x = state.x
        y = state.y
        angle = state.car_angle
        projections = []

        for x_offset, y_offset in self.hiper_parameters.grid_offsets:
            # Calculate projections
            x_proj = np.floor((x - x_offset) / self.field_width)
            y_proj = np.floor((y - y_offset) / self.field_width)
            angle_proj = np.floor((angle + np.pi) / self.delta_car_angle)

            # Clip projections to valid ranges
            x_proj = np.clip(x_proj, 0, self.hiper_parameters.grid_width - 1)
            y_proj = np.clip(y_proj, 0, self.hiper_parameters.grid_height - 1)
            alpha_proj = np.clip(angle_proj, 0, self.hiper_parameters.num_of_car_angle_values - 1)

            # Append the clipped projections to the result
            projections.append((x_proj, y_proj, angle_proj))

        return projections

    def encode_state(self, state: State, action):
        coded_state = np.zeros(shape=self.get_weights_shape())
        projections = self.get_state_projections(state)
        for i, (x_cell_no, y_cell_no, angle_no) in enumerate(projections):
            coded_state[i, int(x_cell_no), int(y_cell_no), int(angle_no), action] = 1.0
        return coded_state.reshape(-1)

class StateStagnationHandler(object):
    closest_distance = None
    def __init__(self, global_variables):
        self.closest_distance = None
        self.smallest_angle = None
        self.max_distance_squared = (global_variables.street_width * global_variables.street_width) + (global_variables.street_length * global_variables.street_length)

    def get_reward_relative_to_closest_distance_achieved(self, state: State):
        if self.closest_distance is None:
            self.closest_distance = get_distance_from_parking(state)
            return 0
        else:
            reward = self.closest_distance - get_distance_from_parking(state)
            self.closest_distance = min(get_distance_from_parking(state), self.closest_distance)
            return reward

    def get_reward_relative_to_smallest_angle_achieved(self, state: State):
        if self.smallest_angle is None:
            self.smallest_angle = state.car_angle
            return 0
        else:
            reward = self.smallest_angle - min(abs(state.car_angle), abs(np.pi - state.car_angle))
            self.smallest_angle = min(min(abs(state.car_angle), abs(np.pi - state.car_angle)), self.smallest_angle)
            return reward

class Linear_Approximator(object):
    encoder = None
    weights = None
    def __init__(self, weights, encoder: CoverageEncoder):
        self.weights = weights
        self.encoder = encoder

    @staticmethod
    def approximate(weights, coded_state):
        return np.sum(np.multiply(weights, coded_state))

def get_distance_from_parking(state: State) -> float:
    return np.sqrt(state.x * state.x + state.y * state.y)

def should_explore(epsylon):
    return np.random.random() < epsylon

def final_angle_reward(angle):
    max_reward = 50
    # Ensure that the angle is between 0 and 2*pi
    angle = angle % (2 * np.pi)
    # Calculate the distance from the closest angle (0 or pi)
    angle_distance = min(abs(angle - 0), abs(angle - np.pi))

    # Map the distance to a value between 0 and 100 (closer to 0 or pi results in higher values)
    scaled_value = max_reward - (((2 * angle_distance) / np.pi) * max_reward)

    # Ensure the result is between 0 and 100
    return max(0, min(100, scaled_value))

def is_in_parking_place(global_variables, state: State):
    return (-global_variables.place_width / 2.0 < state.x < global_variables.place_width / 2.0
     and -global_variables.park_depth / 2.0 < state.y < global_variables.park_depth / 2)

def final_reward(global_variables, state):
    if is_in_parking_place(global_variables, state):
        return 100.0 + final_angle_reward(state.car_angle)
    else:
        return 0.0

def get_reward(global_variables, state, is_collision, quit, state_stagnation_handler: StateStagnationHandler):
    x = state.x
    y = state.y
    collision_reward = 0

    # distance_reward
    distance_reward = 0.1 * ((1 / get_distance_from_parking(state) + 0.5) - 1)
    best_distance_reward = 5 * state_stagnation_handler.get_reward_relative_to_closest_distance_achieved(state)
    #best_angle_reward = -2 * state_stagnation_handler.get_reward_relative_to_smallest_angle_achieved(state) if is_in_parking_place(global_variables, state) else 0
    angle_reward = ((np.pi / 2) - min(abs(state.car_angle), abs(np.pi - state.car_angle))) * (1 / get_distance_from_parking(state))
    parking_place_reward = 5 if is_in_parking_place(global_variables, state) else 0

    if is_collision:
        collision_reward = -50
    if quit:
        value = final_reward(global_variables, state) + best_distance_reward + parking_place_reward + angle_reward + distance_reward
    else:
        value = best_distance_reward + collision_reward + parking_place_reward + angle_reward + + distance_reward

    return value

def choose_action(state, approximator):
    actions = approximator.encoder.get_actions()
    actions_ratings = []
    for i in range(len(actions)):
        actions_ratings.append(Linear_Approximator.approximate(approximator.weights, approximator.encoder.encode_state(state, i)))
    best_action = actions[np.argmax(actions_ratings)]
    angle, velocity = best_action
    return angle, velocity

def park_test(param_fiz, stanp, aproks):
    pm.park_save("param.txt", param_fiz)
    phist = open('historia_park.txt', 'w')
    liczba_stanow_poczatkowych, lparam = stanp.shape
    sr_suma_nagrod = 0
    liczba_krokow = 0

    for epizod in range(liczba_stanow_poczatkowych):
        state_stagnation_handler = StateStagnationHandler(global_variables=pm.GlobalVar)
        # Wybieramy stan poczatkowy:
        nr_stanup = epizod %  liczba_stanow_poczatkowych
        stan = State(stanp[nr_stanup,:])

        krok = 0
        czy_kolizja = False
        czy_zatrzymanie = False
        suma_nagrod_epizodu = 0
        while czy_zatrzymanie == False:
            krok = krok + 1

            # Wyznaczamy akcje a (kąt + kier. ruchu) w stanie stan zgodnie z wyuczoną strategią:
            kat, V = choose_action(stan, aproks)
            
            # zapis kroku historii:
            #phist.write(str(epizod + 1) + "  " + str(krok) + "  " + str(stan[0]) + "  " + str(stan[1]) + "  " + str(stan[2]) + "  " + str(kat) + "  " + str(V) + "\n")
            phist.write("%d %d %.4f %.4f %.4f %.4f %.4f\n" % ((epizod + 1),krok,stan.x,stan.y,stan.car_angle,kat,V))
            # wyznaczenie nowego stanu:
            nowystan, sr_obrotu, czy_kolizja = pm.model_of_car(stan, kat, V, param_fiz)
            nowystan = State(nowystan)

            if (czy_kolizja)|(krok >= param_fiz.max_number_of_steps):
                czy_zatrzymanie = True

            R = get_reward(param_fiz, nowystan, czy_kolizja, czy_zatrzymanie, state_stagnation_handler)
            suma_nagrod_epizodu += R

            stan = nowystan

        sr_suma_nagrod = sr_suma_nagrod + suma_nagrod_epizodu / liczba_stanow_poczatkowych
        liczba_krokow = liczba_krokow + krok
        print("w %d epizodzie suma nagrod = %g, liczba krokow = %d" %(epizod, suma_nagrod_epizodu, krok))

    print("srednia suma nagrod w epizodzie = %g" % (sr_suma_nagrod))
    print("srednia liczba krokow = %g" % (liczba_krokow/liczba_stanow_poczatkowych))
    phist.close()

def get_best_rating_of_state_actions(encoder, new_state, weights):
    best_rating = max(
        Linear_Approximator.approximate(weights, encoder.encode_state(new_state, i))
        for i in range(len(encoder.get_actions()))
    )
    return best_rating

def update_weights_Q_learning(encoder, state, action, weights, alpha, gamma, reward, new_state):
    coded_state = encoder.encode_state(state, action)
    weights += np.multiply(alpha * (reward + gamma * get_best_rating_of_state_actions(encoder, new_state, weights)
            - Linear_Approximator.approximate(weights, coded_state)), coded_state)
    return weights

def car_parked(state: State):
    return (np.degrees(state.car_angle) <= 15 and get_distance_from_parking(state) <= 0.5)

def park_train():
    num_of_epochs = 2001
    alpha = 0.1  # wsp.szybkosci uczenia(moze byc funkcja czasu)
    epsylon = 1 # wsp.eksploracji(moze byc funkcja czasu)
    epsylon_decay_ratio = 0.0005
    gamma = 0.95

    #init_states = np.array([[9.1, 4.6, 0],[6.3, 5.06, 0],[9.6, 3.15, 0],[7.3, 5.75, 0],[10.1, 6.21, 0]],dtype=float)
    init_states = np.array([[10, 5, 0], [9.5, 4.5, 0], [9, 4, 0], [8.5, 3.5, 0]], dtype=float)
    num_of_initial_states, num_of_parameters = init_states.shape

    global_variables = pm.GlobalVar()     # parametry fizyczne parkingu i pojazdu

    # inicjacja kodowania, wyznaczenie liczby parametrów (wag):
    # ........................................................
    # ........................................................

    hiperparameters = HiperParameters(
        grid_width=10,
        grid_height=5,
        grid_offsets=[(0.0, 0.0), (0.5, 0.5), (-0.5, -0.5)],
        num_of_car_angle_values=9,
        num_of_wheel_angle_values=9,
        velocities=(-global_variables.Vmod, global_variables.Vmod))

    encoder = CoverageEncoder(global_variables=global_variables, hiper_parameters=hiperparameters)

    num_of_weights = encoder.count_weights()
    print(f'weights_num = {num_of_weights}')
    weights = np.zeros(num_of_weights)
    actions = encoder.get_actions()
    print(f'actions = {actions}')

    for epoch in range(num_of_epochs):
        print(f'epoch {epoch}')
        state_stagnation_handler = StateStagnationHandler(global_variables=pm.GlobalVar)
        epsylon -= epsylon_decay_ratio

        # Wybieramy stan poczatkowy:
        initial_state_number = epoch % num_of_initial_states
        state = State(init_states[initial_state_number, :])

        step = 0
        quit = False
        while quit == False:
            step += 1

            # Wyznaczamy akcje a (kąt + kier. ruchu) w stanie stan z uwzględnieniem
            # eksploracji (np. metoda epsylon-zachlanna lub softmax)
            # ........................................................
            # ........................................................
            if should_explore(epsylon):
                selected_action = np.random.randint(0, len(actions) - 1)
            else:
                action_ratings = [
                    Linear_Approximator.approximate(weights, encoder.encode_state(state, i))
                    for i in range(len(actions))
                ]
                selected_action = np.argmax(action_ratings)

            angle, velocity = actions[selected_action]

            # wyznaczenie nowego stanu:
            new_state, useless, is_collision = pm.model_of_car(state, angle, velocity, global_variables)
            new_state = State(new_state)

            if is_collision|(step >= global_variables.max_number_of_steps):
                quit = True

            reward = get_reward(global_variables, new_state, is_collision, quit, state_stagnation_handler)

            # Aktualizujemy wartosci Q dla aktualnego stanu i wybranej akcji:
            # ........................................................
            # ........................................................
            # w = w + ...
            weights = update_weights_Q_learning(encoder, state, selected_action, weights, alpha, gamma, reward, new_state)
            state = new_state

        # co jakis czas test z wygenerowaniem historii do pliku:
        if epoch % 50 == 0:
            print("epoch %d\n" % epoch)
            park_test(global_variables, init_states, Linear_Approximator(weights, encoder))

        if car_parked(state):
            park_test(global_variables, init_states, Linear_Approximator(weights, encoder))
            print("Car has successfully parked!\n")
            break


park_train()


