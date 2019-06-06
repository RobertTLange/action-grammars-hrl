from collections import deque

class MacroBuffer(object):
    def __init__(self, capacity, record_macros=False):
        self.buffer = deque(maxlen=capacity)

    def push(self, ep_id, step, state, action,
             reward, next_state, done, tau, string_act):
        self.buffer.append((ep_id, step, state, action,
                            reward, next_state, done, tau, string_action))

    def sample(self, batch_size):
        if not self.record_macros:
            ep_id, step, state, action, reward, next_state, done, tau, active = zip(*random.sample(self.buffer, batch_size))
            return np.stack(state), action, reward, np.stack(next_state), done

    def __len__(self):
        return len(self.buffer)


def macro_action_exec(ep_id, steps, replay_buffer, macro_buffer, macro, env, GAMMA):
    # Macro is a sequence of strings corresponding to primitive actions
    macro_rew = 0
    for i, string_action in enumerate(macro):
        # Decode string to primitive action
        action = letter_to_action(primitive_action)
        next_obs, rew, done, _  = env.step(action)
        # Push primitive transition to ER Buffer
        replay_buffer.push(ep_id, steps+i+1, obs, action,
                           rew, next_obs, done)
        # Accumulate macro reward
        macro_rew += GAMMA**i * rew

    # Push macro transition to ER Buffer
    macro_buffer.push(ep_id, j+1, obs, action,
                      rew, next_obs, done, tau, primitive_action)
    return next_obs, macro_rew, done, _


def letter_to_action(string_action):
    dic = {"a": 0, "b": 1, "c": 2, "d": 3}
    return dic[string_action]


def action_to_letter(action):
    dic = {0: "a", 1: "b", 2: "c", 3: "d"}
    return dic[action]
