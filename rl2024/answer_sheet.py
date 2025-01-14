
############################################################################################################
##########################            RL2023 Assignment Answer Sheet              ##########################
############################################################################################################

# **PROVIDE YOUR ANSWERS TO THE ASSIGNMENT QUESTIONS IN THE FUNCTIONS BELOW.**

############################################################################################################
# Question 2
############################################################################################################

def question2_1() -> str:
    """
    (Multiple choice question):
    For the Q-learning algorithm, which value of gamma leads to the best average evaluation return?
    a) 0.99
    b) 0.8
    return: (str): your answer as a string. accepted strings: "a" or "b"
    """
    answer = "a"  # TYPE YOUR ANSWER HERE "a" or "b"
    return answer


def question2_2() -> str:
    """
    (Multiple choice question):
    For the First-visit Monte Carlo algorithm, which value of gamma leads to the best average evaluation return?
    a) 0.99
    b) 0.8
    return: (str): your answer as a string. accepted strings: "a" or "b"
    """
    answer = "a"  # TYPE YOUR ANSWER HERE "a" or "b"
    return answer


def question2_3() -> str:
    """
    (Multiple choice question):
    Between the two algorithms (Q-Learning and First-Visit MC), whose average evaluation return is impacted by gamma in
    a greater way?
    a) Q-Learning
    b) First-Visit Monte Carlo
    return: (str): your answer as a string. accepted strings: "a" or "b"
    """
    answer = "b"  # TYPE YOUR ANSWER HERE "a" or "b"
    return answer


def question2_4() -> str:
    """
    (Short answer question):
    Provide a short explanation (<100 words) as to why the value of gamma affects more the evaluation returns achieved
    by [Q-learning / First-Visit Monte Carlo] when compared to the other algorithm.
    return: answer (str): your answer as a string (100 words max)
    """
    answer = "The gaps between the mean values of the Q-Learning and First-Visit Monte Carlo for the different gammas are very similar. The change in the gamma value affects the evaluation of Monte Carlo more because in Frozen Lakes has terminal states. This causes First-Visit Monte Carlo methods to only update the early states that lead directly to these terminal states. The gamma value determines how future rewards are valued relative to immediate rewards. A lower gamma value indicates that future rewards are less valuable, making the agent focusing on the immediate outcomes."  # TYPE YOUR ANSWER HERE (100 words max)
    return answer


############################################################################################################
# Question 3
############################################################################################################

def question3_1() -> str:
    """
    (Multiple choice question):
    In Reinforce, which learning rate achieves the highest mean returns at the end of training?
    a) 2e-2
    b) 2e-3
    c) 2e-4
    return: (str): your answer as a string. accepted strings: "a", "b" or "c"
    """
    answer = "c"  # TYPE YOUR ANSWER HERE "a", "b" or "c"
    return answer


def question3_2() -> str:
    """
    (Multiple choice question):
    When training DQN using a linear decay strategy for epsilon, which exploration fraction achieves the highest mean
    returns at the end of training?
    a) 0.99
    b) 0.75
    c) 0.01
    return: (str): your answer as a string. accepted strings: "a", "b" or "c"
    """
    answer = "c"  # TYPE YOUR ANSWER HERE "a", "b" or "c"
    return answer


def question3_3() -> str:
    """
    (Multiple choice question):
    When training DQN using an exponential decay strategy for epsilon, which epsilon decay achieves the highest
    mean returns at the end of training?
    a) 1.0
    b) 0.5
    c) 1e-5
    return: (str): your answer as a string. accepted strings: "a", "b" or "c"
    """
    answer = "c"  # TYPE YOUR ANSWER HERE "a", "b" or "c"
    return answer


def question3_4() -> str:
    """
    (Multiple choice question):
    What would the value of epsilon be at the end of training when employing an exponential decay strategy
    with epsilon decay set to 1.0?
    a) 0.0
    b) 1.0
    c) epsilon_min
    d) approximately 0.0057
    e) it depends on the number of training timesteps
    return: (str): your answer as a string. accepted strings: "a", "b", "c", "d" or "e"
    """
    answer = "b"  # TYPE YOUR ANSWER HERE "a", "b", "c", "d" or "e"
    return answer


def question3_5() -> str:
    """
    (Multiple choice question):
    What would the value of epsilon be at the end of  training when employing an exponential decay strategy
    with epsilon decay set to 0.95?
    a) 0.95
    b) 1.0
    c) epsilon_min
    d) approximately 0.0014
    e) it depends on the number of training timesteps
    return: (str): your answer as a string. accepted strings: "a", "b", "c", "d" or "e"
    """
    answer = "a"  # TYPE YOUR ANSWER HERE "a", "b", "c", "d" or "e"
    return answer


def question3_6() -> str:
    """
    (Short answer question):
    Based on your answer to question3_5(), briefly  explain why a decay strategy based on an exploration fraction
    parameter (such as in the linear decay strategy you implemented) may be more generally applicable across
    different environments  than a decay strategy based on a decay rate parameter (such as in the exponential decay
    strategy you implemented).
    return: answer (str): your answer as a string (100 words max)
    """
    answer = "Decay strategy based on an exploration fraction is exploring consistently during training allowing more equal spread of epsilon values. In contrast, the exponential decay strategy may not adjust well as it relies on a fixed decay rate, which could lead to either too rapid decrease in exploration in when the decay factor is low or insufficient decrease in higher decay factors. This can result in suboptimal policy learning as the exploration is not as spread as in the linear decay. For example, the 0.95 decay factor restrict the epsilon to be between 1 and 0.95."  # TYPE YOUR ANSWER HERE (100 words max)
    return answer


def question3_7() -> str:
    """
    (Short answer question):
    In DQN, explain why the loss is not behaving as in typical supervised learning approaches
    (where we usually see a fairly steady decrease of the loss throughout training)
    return: answer (str): your answer as a string (150 words max)
    """
    answer = "The loss is not behaving as in typical supervised learning approaches as it is a non-stationary problem, the model learns as it evolves. Additionally, the data contradicts the supervised learning assumption of independent and identically distributed data. The exploration affects the learning process of the network as it introduces variability especially at the beginning stages. Additional variability is introduced by the temporal difference learning. Add of these noises and factors produce atypical loss behaviour."  # TYPE YOUR ANSWER HERE (150 words max)
    return answer


def question3_8() -> str:
    """
    (Short answer question):
    Provide an explanation for the spikes which can be observed at regular intervals throughout
    the DQN training process.
    return: answer (str): your answer as a string (100 words max)
    """
    answer = "The loss is spiking as at every 2000 steps (target_update_freq steps), the targets are updated (a hard update is performed). The target critics gets the critics weights which causes the drastical loss increase. After each update the function is fitted for 2000 steps."  # TYPE YOUR ANSWER HERE (100 words max)
    return answer


############################################################################################################
# Question 5
############################################################################################################

def question5_1() -> str:
    """
    (Short answer question):
    Provide a short description (200 words max) describing your hyperparameter turning and scheduling process to get
    the best performance of your agents
    return: answer (str): your answer as a string (200 words max)
    """
    answer = "Initially, I started by using the tuned layer sises from exercise 4 as a starting point. For the learning rates I explored 3 parameters (1e-2, 1e-3 and 1e-4) as they are very common well performing metrics. For the gamma and tau I used a random search (choosing 3 values) between the values (0.8 - 0.995) and (0.1 - 0.9) respectively. I tried 5 different batch sizes: 8, 16, 32, 16, 64 and kept the default buffer_capacity. All of the chosen specific values was selected as many of the models I researched use I of the given values. Based on the results of the best performing combination [policy_learning_rate, critic_learning_rate, critic_hidden_size, policy_hidden_size, gamma, tau, batch_size, buffer_capacity] = [1e-2, 1e-2, [65,65,65], [105,105,105], 0.95, 0.6, 64, 1e7], I continued with a grid search with values around this combination (increasing gamma and tau with 0.1, 0.2 floats and the learning rates with very small values). In the end, I found that the combination mention above achieves the best training mean (of around 1418) and the most stable evaluation - achieving mean value of over 800 in around 90% of the evaluations."  # TYPE YOUR ANSWER HERE (200 words max)
    return answer