"""
Unit tests for Graded Exercise #2
"""

import numpy as np
from os import path
from .helpers import questions


# Hyperparameters of the tests. Also used for test data generation
# Register all tests here!

test_params = {
    'normalization': dict(data_path="normalization.npz", N=100, dims=3),
    'feature_expansion': dict(data_path="feature_expansion.npz", N=100, dims=3, degree=7),
    'loss_function': dict(data_path="loss_function.npz", N=100, dims=3, lmbda=0.1),
    'gradient': dict(data_path="gradient.npz", N=100, dims=3, lmbda=0.1),
    'training_with_gd': dict(data_path="training_with_gd.npz", N=100, dims=3, epochs=50, lr=1e-3, lmbda=1e-2),
    'cross_validation': dict(data_path="cross_validation.npz", N=100, dims=3, num_folds=5, lmbda=1e-2, degree=7, epochs=1000, lr=1e-2),
    'error_values': dict(data_path="error_values.npz", N=100, dims=3, num_folds=5, lmbdas=np.logspace(-5,-2,3), degrees=np.arange(1,8,3)),
    'hyperparameters': dict(data_path="hyperparameters.npz", lmbdas=np.logspace(-5,-2,3), degrees=np.arange(1,8,3)),
    'final_training': dict(data_path="final_training.npz", N=100, dims=3, lmbda=1e-3, degree=4, epochs=500, lr=1e-2),
}


test_data_path = path.join('lib', 'tests_data')


def get_test_data(test_id):
    """
    Loads data required by the test passed as parameter. Test must be registered in test_params dictionary above.
    """
    test_data = None
    test_hyperparams = test_params.get(test_id)

    data_path = test_hyperparams.get("data_path")
    if data_path is None:
        return None

    with np.load(path.join(test_data_path, data_path)) as test_data_file:
        test_data = dict(test_data_file.items())

    return test_data


def test(test_function):
    """
    Decorator: pretty-prints handled exceptions, raises others so we can write tests for those.
    """
    def process_exceptions(*args, **kwargs):
        exception_caught = None

        function_result = None

        try:
            function_result = test_function(*args, **kwargs)
        except AssertionError as e:
            exception_caught = e
        except Exception as other_exception:
            raise other_exception

        if exception_caught is None:
            print("[{}] - No problems detected. Your code is correct! \U0001f60e".format(test_function.__name__))
        else:
            print("[{}] - {} \U0001f635".format(test_function.__name__, exception_caught))

        return function_result

    return process_exceptions


class isolatedRNG:
    """
    Use fixed random seed locally.
    It saves and restores Random Number Generator state.

    Use as context guard with 'with' statement as
    with isolatedRNG():
        do your random thing
    """
    def __init__(self, seed=28008):
        self.seed = seed
        self.rng_state = None

    def __enter__(self):
        self.rng_state = np.random.get_state()
        np.random.seed(self.seed)

    def __exit__(self, type, value, traceback):
        np.random.set_state(self.rng_state)


#############################
#     Utility functions     #
#############################

def resolve(var_name, scope):
    """
    Finds the given symbol (function or variable) in the scope. Reports error if not found.
    """
    variable = scope.get(var_name)
    fail_msg = "'{}' is not defined in the current scope".format(var_name)
    assert variable is not None, fail_msg
    return variable


def get_submission_path(scope):
    """
    Returns the filename for the submission for the current student.
    Also validates the student's sciper number.
    """
    sciper_number = resolve('sciper_number', scope)

    # ensure sciper_number has right format
    sciper_number = str(sciper_number)
    fail_msg = "Please correct your sciper number, otherwise you will not be able to generate your results file"
    assert str.isdigit(sciper_number) and len(sciper_number) == len('123456'), fail_msg

    answers_file_path = "answers_{}.npz".format(sciper_number)

    return answers_file_path


def register_answer(name, answer, scope, save=False):
    """
    Add answer to answers dictionary for a test. Make sure names are unique.
    """
    answers_file_path = get_submission_path(scope)

    if 'answers' not in scope:
        scope['answers'] = dict()
        if path.isfile(answers_file_path):
            scope['answers'].update(np.load(answers_file_path).items())
    scope.get('answers')[name] = answer

    np.savez(answers_file_path, **scope.get('answers'))


def compare_np_arrays(computed, expected, *, compare_types=True, compare_shapes=True, compare_values=True, varname=None):
    """
    Common function to compare two numpy arrays.
    - Validates that computed object is a numpy array.
    - Compares computed and expected types.
    - Compares computed and expected sizes.
    - Compares computed and expected values.
    """
    # Ensure computed is array
    fail_msg = "Computed {} is not a numpy array. It's of type {}".format(varname or 'object', type(computed))
    assert isinstance(computed, np.ndarray), fail_msg

    # Ensure arrays are same type
    if compare_types:
        fail_msg = "Computed {} is of type {}, but {} was expected".format(varname or 'array', computed.dtype, expected.dtype)
        assert computed.dtype == expected.dtype, fail_msg

    # Ensure arrays are same shape
    if compare_shapes:
        fail_msg = "Computed {} dimensions are {}, but {} was expected".format(varname or 'array', computed.shape, expected.shape)
        assert computed.shape == expected.shape, fail_msg

    # Ensure arrays have similar values
    if compare_values:
        fail_msg = "Computed {} does not have the expected values. Make sure your computation is correct.".format(varname or 'array')
        assert np.all(np.isclose(computed, expected, atol=1e-3)), fail_msg



#############################
#           Tests           #
#############################

@test
def test_normalization(scope):
    test_id = "normalization"

    # Test data
    test_data = get_test_data(test_id)
    X = test_data.get('X')
    expected_mean = test_data.get('mu')
    expected_std = test_data.get('std')
    expected_stats = (expected_mean, expected_std)
    print("[test_normalization] Testing normalization on random data (N={}, d={})...".format(*X.shape))

    # [TEST] required functions exist
    find_stats = resolve('find_stats', scope)
    normalize = resolve('normalize', scope)

    # Apply student's find_stats
    computed_stats = find_stats(X)
    # Save computed_stats
    register_answer('find_stats_result', computed_stats, scope)

    # [TEST] 'find_stats' returns two values
    if computed_stats is None:
        n_return = 0
    elif not isinstance(computed_stats, tuple):
        n_return = 1
    else:
        n_return = len(computed_stats)
    fail_msg = "Your function should return {} values but returns {} instead".format(len(expected_stats), n_return)
    assert len(expected_stats) == n_return, fail_msg

    computed_mean, computed_std = computed_stats

    # [TEST] mean and std are right type, shape, and value
    compare_np_arrays(computed_mean, expected_mean, varname="mean")
    compare_np_arrays(computed_std, expected_std, varname="std")

    # Apply student's normalize
    normalized_X = normalize(X, computed_mean, computed_std)
    # Save computed normalized matrix
    register_answer('normalize_result', normalized_X, scope)

    # [TEST] 'normalize' returns array of same shape
    fail_msg = "Your normalized matrix should be the same dimensions as the original"
    compare_np_arrays(normalized_X, X, compare_values=False, varname="normalized data")

    # [TEST] 'normalize' returns normalized array
    fail_msg = "All features in your normalized data should have mean 0"
    assert np.isclose(normalized_X.sum(), 0), fail_msg
    fail_msg = "All features in your normalized data should have standard deviation 1"
    assert np.isclose(normalized_X.std(), 1), fail_msg

    # [TEST] computed stats for normalized vector are 0, 1
    stats_norm = find_stats(normalized_X)
    fail_msg =  "Your find_stats method does not compute the right statistics"
    assert np.all(np.isclose(stat, expected_stat) for stat, expected_stat in zip(stats_norm, (0, 1))), fail_msg


@test
def test_feature_expansion(scope):
    test_id = "feature_expansion"

    # Test hyperparameters
    test_hyperparams = test_params.get(test_id)
    degree = test_hyperparams.get('degree')

    # Test data
    test_data = get_test_data(test_id)
    X = test_data.get("X")
    expected_expanded_X = test_data.get("expanded_X")

    print("[test_feature_expansion] Testing degree {} polynomial feature expansion on random data (N={}, d={})...".format(degree, *X.shape))

    # [TEST] 'expand' function exists
    expand = resolve('expand', scope)

    # Apply student's expand
    computed_expanded_X = expand(X, degree=degree)
    # Save computed expanded X
    register_answer('expand_result', computed_expanded_X, scope)

    # [TEST] computed feature expansion is right type, shape, and value
    compare_np_arrays(computed_expanded_X, expected_expanded_X, varname="feature expansion")


@test
def test_loss_function(scope):
    test_id = "loss_function"

    # Test hyperparameters
    test_hyperparams = test_params.get(test_id)
    lmbda = test_hyperparams.get('lmbda')

    # Test data
    test_data = get_test_data(test_id)
    X = test_data.get("X")
    y = test_data.get("y")
    w = test_data.get("w")
    expected_loss_value = test_data.get("loss_value")
    expected_loss_value = expected_loss_value.item()  # Because scalars are also saved as arrays

    # [TEST] 'loss' function exists
    loss = resolve('loss', scope)

    # Apply student's loss
    computed_loss_value = loss(X, y, w, lmbda=lmbda)
    # Save computed loss value
    register_answer('loss_value_result', computed_loss_value, scope)

    # [TEST] computed loss is right type
    fail_msg = "Your computed loss is of type {} but it should be {}".format(type(computed_loss_value), type(expected_loss_value))
    assert isinstance(computed_loss_value, type(expected_loss_value)), fail_msg

    # [TEST] computed loss is right value
    fail_msg = "Computed loss does not have the expected value. Make sure your computation is correct."
    assert np.isclose(computed_loss_value, expected_loss_value), fail_msg


@test
def test_gradient(scope):
    test_id = "gradient"

    # Test hyperparameters
    test_hyperparams = test_params.get(test_id)
    lmbda = test_hyperparams.get("lmbda")

    # Test data
    test_data = get_test_data(test_id)
    X = test_data.get("X")
    y = test_data.get("y")
    w = test_data.get("w")
    expected_gradient_value = test_data.get("gradient_value")

    # [TEST] 'gradient' function exists
    gradient = resolve('gradient', scope)

    # Apply student's gradient
    computed_gradient_value = gradient(X, y, w, lmbda=lmbda)
    # Save computed gradient value
    register_answer('gradient_value_result', computed_gradient_value, scope)

    # [TEST] computed gradient is right type, shape, and value
    compare_np_arrays(computed_gradient_value, expected_gradient_value, varname="gradient")


@test
def test_gd_training(scope):
    test_id = "training_with_gd"

    # Test hyperparameters
    test_hyperparams = test_params.get(test_id)
    epochs = test_hyperparams.get("epochs")
    lr = test_hyperparams.get("lr")
    lmbda = test_hyperparams.get("lmbda")

    # Test data
    test_data = get_test_data(test_id)
    X = test_data.get("X_train")
    y = test_data.get("y_train")
    expected_weights = test_data.get("weight")

    # [TEST] 'training_with_gd' function exists
    training_with_gd = resolve('training_with_gd', scope)

    # Apply student's training_with_gd
    with isolatedRNG():
        computed_weights = training_with_gd(X, y, epochs, lr, lmbda, debug_printing_on=False)
    # Save computed model parameters
    register_answer('gradient_descent_result', computed_weights, scope)

    # [TEST] model parameter vector is right type, shape, and value
    compare_np_arrays(computed_weights, expected_weights, varname="model weight vector")


@test
def test_cross_validation(scope):
    test_id = "cross_validation"

    # Test hyperparameters
    test_hyperparams = test_params.get(test_id)
    num_folds = test_hyperparams.get('num_folds')
    lmbda = test_hyperparams.get('lmbda')
    degree = test_hyperparams.get('degree')
    epochs = test_hyperparams.get('epochs')
    lr = test_hyperparams.get('lr')

    # Test data
    test_data = get_test_data(test_id)
    X = test_data.get("X")
    y = test_data.get("y")
    k_fold_indices = test_data.get("k_fold_indices")
    expected_error = test_data.get("error")

    # [TEST] 'run_cross_validation' function exists
    run_cross_validation = resolve('run_cross_validation', scope)

    # Apply student's run_cross_validation
    with isolatedRNG():
        predicted_error = run_cross_validation(k_fold_indices, num_folds, X, y, lmbda, degree, epochs, lr)
    # Save computed predicted error
    register_answer('cross_validation_result', predicted_error, scope)

    # [TEST] predicted error has right value
    fail_msg = "Computed error does not have the expected value. Make sure your computation is correct."
    assert np.isclose(predicted_error, expected_error, atol=1e-3), fail_msg


@test
def test_error_values(scope):
    test_id = "error_values"

    # Test hyperparameters
    test_hyperparams = test_params.get(test_id)
    num_folds = test_hyperparams.get('num_folds')
    lmbdas = test_hyperparams.get('lmbdas')
    degrees = test_hyperparams.get('degrees')

    # Test data
    test_data = get_test_data(test_id)
    X = test_data.get("X")
    y = test_data.get("y")
    expected_error_values = test_data.get("error_values")

    # [TEST] 'grid_search_for_hyperparameters' function exists
    grid_search_for_hyperparameters = resolve('grid_search_for_hyperparameters', scope)

    # Apply student's grid_search_for_hyperparameters
    error_values = grid_search_for_hyperparameters(X, y, num_folds, lmbdas, degrees)

    # Save computed error values
    register_answer('error_values_result', error_values, scope)

    # [TEST] error has right value
    compare_np_arrays(error_values, expected_error_values)


@test
def test_hyperparameters(scope):
    test_id = "hyperparameters"

    # Test hyperparameters
    # test_hyperparams = test_params.get(test_id)
    # lmbdas = test_hyperparams.get('lmbdas')
    # degrees = test_hyperparams.get('degrees')
    # Loading from scope now, apparently...
    lambdas = resolve("lambdas", scope)
    degrees = resolve("degrees", scope)
    error_values = resolve("error_values", scope)

    # error_values = test_data.get("error_values")
    # Test data
    test_data = get_test_data(test_id)
    expected_lambda_best = test_data.get("lambda_best")
    expected_degree_best = test_data.get("degree_best")

    find_best_hyperparameters = resolve('find_best_hyperparameters', scope)

    # Apply student's find_best_hyperparameters
    lambda_best, degree_best = find_best_hyperparameters(error_values, lambdas, degrees)

    # Save computed hyperparameters
    register_answer('hyperparameters_result', (lambda_best, degree_best), scope)

    # [TEST] deg_best maps to same one-way function value
    fail_msg = "Wrong value for lambda_best"
    assert lambda_best == expected_lambda_best, fail_msg

    # [TEST] lambda_best maps to same one-way function value
    fail_msg = "Wrong value for degree_best"
    assert degree_best == expected_degree_best, fail_msg


@test
def test_train_and_predict_final_model(scope):
    test_id = "final_training"

    # Test hyperparameters
    test_hyperparams = test_params.get(test_id)
    lmbda = test_hyperparams.get('lmbda')
    degree = test_hyperparams.get('degree')
    epochs = test_hyperparams.get('epochs')
    lr = test_hyperparams.get('lr')

    # Test data
    test_data = get_test_data(test_id)
    X = test_data.get("X")
    y = test_data.get("y")
    X_test = test_data.get("X_test")
    w_end_expected = test_data.get("w_end")
    y_pred_expected = test_data.get("y_pred")

    # [TEST] 'train_and_predict_final_model' function exists
    train_and_predict_final_model = scope.get('train_and_predict_final_model')

    # Apply student's train_and_predict_final_model
    w_end, y_pred = train_and_predict_final_model(X, y, X_test, lmbda, degree, epochs=epochs, lr=lr)

    # Save computed model parameters and predictions
    register_answer('model_parameters_result', w_end, scope)
    register_answer('predictions', y_pred, scope)

    compare_np_arrays(w_end, w_end_expected)
    compare_np_arrays(y_pred, y_pred_expected)


@test
def test_theory_answers(scope):
    test_id = "theory"

    # [TEST] 'theory_answers' variable exists
    theory_answers = resolve('theory_answers', scope)

    theory_answers.extend(['X'] * (len(questions) - len(theory_answers)))

    # Save provided theory answers
    theory_answers = [str(answer).upper() for answer in theory_answers]
    register_answer('theory', theory_answers, scope)

    # [TEST] 'theory_answers' is of right shape
    assert len(theory_answers) == 4, "You do not have the correct number of answers"

    fail_msg = "Invalid answer. Must answer either T or F (or X if you are skipping a question)"
    assert all(answer in ('T', 'F', 'X') for answer in theory_answers), fail_msg
