import time
import random


class ExpertNoiser(object):

    # define frequency into noise events per minute
    # define the amount_of_time of setting the noise

    def __init__(self, noise_type, frequency=15, intensity=10, min_noise_time_amount=2.0):

        # self.variable_type = variable
        self.noise_type = noise_type
        self.frequency = frequency
        self.noise_being_set = False
        self.noise_start_time = time.time()
        self.noise_end_time = time.time() + 1
        self.min_noise_time_amount = min_noise_time_amount
        self.noise_time_amount = min_noise_time_amount + float(random.randint(50, 200) / 100.0)
        self.second_counter = time.time()
        self.steer_noise_time = 0
        self.intensity = intensity + random.randint(-2, 2)
        self.remove_noise = False
        self.current_noise_mean = 0

    def set_noise(self):

        if self.noise_type == 'Spike' or self.noise_type == 'Throttle':

            # flip between positive and negative
            coin = random.randint(0, 1)
            if coin == 0:  # negative
                self.current_noise_mean = 0.001  # -random.gauss(0.05,0.02)
            else:  # positive
                self.current_noise_mean = -0.001  # random.gauss(0.05,0.02)

    def get_noise(self):

        if self.noise_type == 'Spike' or self.noise_type == 'Throttle':
            if self.current_noise_mean > 0:

                return min(0.55,
                           self.current_noise_mean + (
                               time.time() - self.noise_start_time) * 0.03 * self.intensity)
            else:

                return max(-0.55,
                           self.current_noise_mean - (
                               time.time() - self.noise_start_time) * 0.03 * self.intensity)

    def get_noise_removing(self):
        # print 'REMOVING'
        added_noise = (self.noise_end_time - self.noise_start_time) * 0.02 * self.intensity
        # print added_noise
        if self.noise_type == 'Spike' or self.noise_type == 'Throttle':
            if self.current_noise_mean > 0:
                added_noise = min(0.55, added_noise + self.current_noise_mean)
                return added_noise - (time.time() - self.noise_end_time) * 0.03 * self.intensity
            else:
                added_noise = max(-0.55, self.current_noise_mean - added_noise)
                return added_noise + (time.time() - self.noise_end_time) * 0.03 * self.intensity

    def is_time_for_noise(self, steer):

        # Count Seconds
        second_passed = False
        if time.time() - self.second_counter >= 1.0:
            second_passed = True
            self.second_counter = time.time()

        if time.time() - self.noise_start_time >= self.noise_time_amount and not self.remove_noise and self.noise_being_set:
            self.noise_being_set = False
            self.remove_noise = True
            self.noise_end_time = time.time()

        if self.noise_being_set:
            return True

        if self.remove_noise:
            # print "TIME REMOVING ",(time.time()-self.noise_end_time)
            if (time.time() - self.noise_end_time) > (
                    self.noise_time_amount):  # if half the amount put passed
                self.remove_noise = False
                self.noise_time_amount = self.min_noise_time_amount + float(
                    random.randint(50, 200) / 100.0)
                return False
            else:
                return True

        if second_passed and not self.noise_being_set:
            # Ok, if noise is not being set but a second passed... we may start puting more

            seed = random.randint(0, 60)
            if seed < self.frequency:
                if not self.noise_being_set:
                    self.noise_being_set = True
                    self.set_noise()
                    self.steer_noise_time = steer
                    self.noise_start_time = time.time()
                return True
            else:
                return False

        else:
            return False

    def set_noise_exist(self, noise_exist):
        self.noise_being_set = noise_exist

    def compute_noise(self, action, speed):

        # noisy_action = action
        if self.noise_type == 'None':
            return action, False, False

        if self.noise_type == 'Spike':

            if self.is_time_for_noise(action.steer):
                steer = action.steer

                if self.remove_noise:
                    steer_noisy = max(
                        min(steer + self.get_noise_removing() * (25 / (2.3 * speed + 5)), 1), -1)

                else:
                    steer_noisy = max(min(steer + self.get_noise() * (25 / (2.3 * speed + 5)), 1),
                                      -1)

                noisy_action = action

                noisy_action.steer = steer_noisy

                return noisy_action, False, not self.remove_noise

            else:
                return action, False, False

        if self.noise_type == 'Throttle':
            if self.is_time_for_noise(action.throttle):
                throttle_noisy = action.throttle
                brake_noisy = action.brake

                if self.remove_noise:
                    # print(" Throttle noise removing", self.get_noise_removing())
                    noise = self.get_noise_removing()
                    if noise > 0:
                        throttle_noisy = max(min(throttle_noisy + noise, 1), 0)
                    else:
                        brake_noisy = max(min(brake_noisy + -noise, 1), 0)

                else:

                    # print(" Throttle noise ", self.get_noise())
                    noise = self.get_noise()
                    if noise > 0:
                        throttle_noisy = max(min(throttle_noisy + noise, 1), 0)
                    else:
                        brake_noisy = max(min(brake_noisy + -noise, 1), 0)

                noisy_action = action
                noisy_action.throttle = throttle_noisy
                noisy_action.brake = brake_noisy

                # print 'timefornosie'
                return noisy_action, False, not self.remove_noise
            else:
                return action, False, False
