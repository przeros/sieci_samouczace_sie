import numpy as np
import tensorflow as tf
import parking_model as pm
from functools import reduce
from keras.losses import MSE
from keras.models import Sequential
from keras.layers import Dense, InputLayer, Dropout
from tqdm import trange
import random

def predict(state, approximator: Sequential):
    return approximator.predict(np.array([state]), verbose=0)[0]

class State:
    def __init__(self, state_vector: list):
        self.x = state_vector[0]
        self.y = state_vector[1]
        self.car_angle = state_vector[2]

class HiperParameters(object):
   def __init__(self, num_of_car_angle_values, num_of_wheel_angle_values, velocities, num_of_prototypes, r):
       self.num_of_car_angle_values = num_of_car_angle_values
       self.grid_num_of_wheel_angle_values = num_of_wheel_angle_values
       self.velocities = velocities
       self.num_of_prototypes = num_of_prototypes
       self.r = r

class PrototypeEncoder(object):
    def __init__(self, global_variables: pm.GlobalVar, hiper_parameters: HiperParameters):
        self.global_variables = global_variables
        self.hiper_parameters = hiper_parameters
        self.delta_car_angle = 2.0 * np.pi / hiper_parameters.num_of_car_angle_values
        self.car_angles = np.arange(-np.pi, np.pi, self.delta_car_angle)
        self.delta_wheel_angle = 2.0 * global_variables.wheel_turn_angle_max / hiper_parameters.grid_num_of_wheel_angle_values
        self.wheel_angles = np.arange(-global_variables.wheel_turn_angle_max, global_variables.wheel_turn_angle_max, self.delta_wheel_angle)
        self.prototypes = self.generate_prototypes()
        print('car_angles =', self.car_angles)
        print('wheel_angles =', self.wheel_angles)

    def count_weights(self):
        return reduce(lambda x, y: x * y, self.get_weights_shape(), 1)

    def get_weights_shape(self) -> [int]:
        return [
            self.hiper_parameters.num_of_prototypes,
            len(self.get_actions())
        ]

    def get_actions(self) -> [(float, float)]:
        wheel_turn_actions = np.tile(self.wheel_angles, len(self.hiper_parameters.velocities))
        velocity_actions = np.repeat(self.hiper_parameters.velocities, len(self.wheel_angles))
        return np.column_stack((wheel_turn_actions, velocity_actions))

    def generate_prototypes(self):
        prototypes = np.random.rand(self.hiper_parameters.num_of_prototypes, 3)
        prototypes[:, 0] *= self.global_variables.street_length
        prototypes[:, 1] *= self.global_variables.street_width
        prototypes[:, 2] = np.random.uniform(-np.pi, np.pi, self.hiper_parameters.num_of_prototypes)
        return prototypes

    # def get_state_projections(self, state: State):
    #     distances = np.linalg.norm(self.prototypes - (state.x, state.y, state.car_angle), axis=1)
    #     close_indices = np.where(distances <= self.hiper_parameters.r)[0]
    #     return close_indices

    # def encode_state(self, state: State, action):
    #     coded_state = np.zeros(shape=self.get_weights_shape())
    #     projections = self.get_state_projections(state)
    #     for projection in projections:
    #         coded_state[projection, action] = 1.0
    #     return coded_state.reshape(-1)

class StateStagnationHandler(object):
    closest_distance = None
    def __init__(self, global_variables):
        self.closest_distance = None
        self.smallest_angle = None
        self.max_distance_squared = (global_variables.street_width * global_variables.street_width) \
                                    + (global_variables.street_length * global_variables.street_length)

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

# class Linear_Approximator(object):
#     encoder = None
#     weights = None
#     def __init__(self, weights, encoder: PrototypeEncoder):
#         self.weights = weights
#         self.encoder = encoder
#
#     @staticmethod
#     def approximate(weights, coded_state):
#         return np.sum(np.multiply(weights, coded_state))

class ReplayBuffer:
    def __init__(self, max_size):
        self.buffer = []
        self.max_size = max_size
        self.next_idx = 0

    def add(self, experience):
        if len(self.buffer) < self.max_size:
            self.buffer.append(experience)
        else:
            self.buffer[self.next_idx] = experience
            self.next_idx = (self.next_idx + 1) % self.max_size

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

def get_distance_from_parking(state: State) -> float:
    return np.sqrt(state.x * state.x + state.y * state.y)

def exploration(epsylon):
    return np.random.random() < epsylon

def get_final_score(physical_parameters, state: (float, float, float)) -> float:
    if (-physical_parameters.place_width / 2.0 < state.x < physical_parameters.place_width / 2.0
            and -physical_parameters.park_depth / 2.0 < state.y < physical_parameters.park_depth / 2.0):
        return 100.0
    return 0.0

def get_reward(physical_parameters, state, is_collision, is_stopped, recorder):
    value = 0
    x = state.x
    y = state.y
    alpha = state.car_angle
    xy_distance_squared = x * x + y * y

    alpha_reduced = 0
    if physical_parameters.if_side_parking_place:
        if np.abs(alpha) > np.pi / 2:
            alpha_reduced = np.pi - np.abs(alpha)
        else:
            alpha_reduced = np.abs(alpha)
    else:
        alpha_reduced = np.abs(np.abs(alpha) - np.pi / 2)

    alpha_reduced = alpha_reduced / (xy_distance_squared + 0.5)

    # distance_reward = 1 / (xy_distance_squared + 0.5) - 1
    distance_reward = recorder.get_reward_relative_to_closest_distance_achieved(state)
    if distance_reward > 0.0:
        distance_reward *= (
                (physical_parameters.max_number_of_steps) / (
                    physical_parameters.max_number_of_steps + 1)
        ) * 3.0
        if recorder.closest_distance is not None and recorder.closest_distance < 5.0:
            distance_reward *= 2.0
        if recorder.closest_distance is not None and recorder.closest_distance < 2.5:
            distance_reward *= 4.0
    alpha_reward = alpha_reduced - 0.5

    # jeśli V==0 nagroda na podstawie odległości

    if is_collision:
        value = -10.0
    elif is_stopped:
        value = distance_reward + (4 * alpha_reward) + get_final_score(physical_parameters, state)
    else:
        value = distance_reward

    return value

def choose_action(state, approximator: Sequential, encoder):
    actions = encoder.get_actions()
    actions_ratings = predict(np.array([state.x, state.y, state.car_angle]), approximator)
    best_action = actions[np.argmax(actions_ratings)]
    angle, velocity = best_action
    return angle, velocity

def park_test(param_fiz, stanp, approximator: Sequential, encoder):
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
            kat, V = choose_action(stan, approximator, encoder)
            
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
        print("w %d epizodzie suma nagrod = %g, liczba krokow = %d" %(epizod+1, suma_nagrod_epizodu, krok))

    print("srednia suma nagrod w epizodzigrue = %g" % (sr_suma_nagrod))
    print("srednia liczba krokow = %g" % (liczba_krokow/liczba_stanow_poczatkowych))
    phist.close()

def get_best_rating_of_state_actions(approximator: Sequential, new_state):
    best_rating = max(predict(np.array([new_state.x, new_state.y, new_state.car_angle]), approximator))
    return best_rating


def update_model_Q_learning_minibatch(buffer, batch_size, deep_nn_approximator, encoder, gamma):
    minibatch = buffer.sample(batch_size)
    for experience in minibatch:
        state, action, reward, next_state, done = experience

        # Przetwarzanie stanów i akcji
        current_state_input = np.array([state.x, state.y, state.car_angle])
        next_state_input = np.array([next_state.x, next_state.y, next_state.car_angle])

        # Obliczanie docelowych wartości Q
        target_q = reward
        if not done:
            target_q += gamma * get_best_rating_of_state_actions(deep_nn_approximator, next_state)

        # Aktualizacja sieci neuronowej
        with tf.GradientTape() as tape:
            current_q_values = deep_nn_approximator(np.array([current_state_input]))
            loss = MSE([target_q], [current_q_values[0, action]])
        grads = tape.gradient(loss, deep_nn_approximator.trainable_variables)
        deep_nn_approximator.optimizer.apply_gradients(zip(grads, deep_nn_approximator.trainable_variables))

    return deep_nn_approximator

def car_parked(state: State):
    return (np.degrees(state.car_angle) <= 15 and get_distance_from_parking(state) <= 0.5)

def park_train():
    num_of_epochs = 300
    alpha = 0.1  # wsp.szybkosci uczenia(moze byc funkcja czasu)
    epsylon = 0 # wsp.eksploracji(moze byc funkcja czasu)
    epsylon_decay_ratio = 0
    gamma = 0.95
    lambda_coeff = 0.3

    buffer = ReplayBuffer(max_size=1000)
    batch_size = 20

    #init_states = np.array([[9.1, 4.6, 0],[6.3, 5.06, 0],[9.6, 3.15, 0],[7.3, 5.75, 0],[10.1, 6.21, 0]],dtype=float)
    init_states = np.array([[10, 5, 0], [9.5, 4.5, 0], [9, 4, 0], [8.5, 3.5, 0]], dtype=float)
    #init_states = np.array([[10, 5, 0]], dtype=float)
    num_of_initial_states, num_of_parameters = init_states.shape

    global_variables = pm.GlobalVar()     # parametry fizyczne parkingu i pojazdu

    # inicjacja kodowania, wyznaczenie liczby parametrów (wag):
    # ........................................................
    # ........................................................

    hiperparameters = HiperParameters(
        num_of_car_angle_values=9,
        num_of_wheel_angle_values=9,
        velocities=(-global_variables.Vmod, global_variables.Vmod),
        num_of_prototypes=2000,
        r=1.0
    )

    encoder = PrototypeEncoder(global_variables=global_variables, hiper_parameters=hiperparameters)

    num_of_weights = encoder.count_weights()
    print(f'weights_num = {num_of_weights}')
    weights = np.zeros(num_of_weights)
    actions = encoder.get_actions()
    print(f'actions = {actions}')

    # Initialize the neural network approximator
    input_shape = (3,)
    output_shape = len(actions)
    deep_nn_approximator = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_shape=input_shape),
        # czy tu powinien byc relu?
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(output_shape, activation='softmax')
    ])
    deep_nn_approximator.compile(optimizer='adam', loss='mse')
    deep_nn_approximator.summary()

    epochs = trange(num_of_epochs + 1, desc='episodes', leave=True)
    for epoch in epochs:
        state_stagnation_handler = StateStagnationHandler(global_variables=pm.GlobalVar)
        epsylon -= epsylon_decay_ratio

        # Inicjujemy wektor z i wybieramy stan poczatkowy:
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
            if exploration(epsylon):
                selected_action = np.random.randint(0, len(actions) - 1)
            else:
                action_ratings = predict(np.array([state.x, state.y, state.car_angle]), deep_nn_approximator)

                selected_action = np.argmax(action_ratings)

            angle, velocity = actions[selected_action]

            # wyznaczenie nowego stanu:
            new_state, useless, is_collision = pm.model_of_car(state, angle, velocity, global_variables)
            new_state = State(new_state)

            if is_collision|(step >= global_variables.max_number_of_steps):
                quit = True

            reward = get_reward(global_variables, new_state, is_collision, quit, state_stagnation_handler)

            buffer.add((state, selected_action, reward, new_state, quit))

            # Aktualizujemy wartosci Q dla aktualnego stanu i wybranej akcji:
            # ........................................................
            # ........................................................
            # w = w + ...
            if len(buffer.buffer) > batch_size:
                deep_nn_approximator = update_model_Q_learning_minibatch(buffer, batch_size, deep_nn_approximator, encoder, gamma)

            state = new_state

        # co jakis czas test z wygenerowaniem historii do pliku:
        if epoch % 5 == 0:
            print("epoch %d\n" % epoch)
            park_test(global_variables, init_states, deep_nn_approximator, encoder)

        if car_parked(state):
            park_test(global_variables, init_states, deep_nn_approximator, encoder)
            print("Car has successfully parked!\n")
            break

park_train()



