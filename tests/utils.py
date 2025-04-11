import numpy as np


def get_call_expected_dims(n_samples: int, is_old_api: bool):
    if is_old_api:
        return (n_samples, )
    return (n_samples, 1)


def get_sample_expected_dims(n_samples: int, n_obs: int, is_old_api: bool):
    return (n_samples * n_obs, 1)


def check_func_call_dims(func, func_log, is_old_api=False):
    x1 = np.random.rand(17, func.dim)
    x2 = np.random.rand(func.dim, func.dim)
    x3 = np.random.rand(1, func.dim)

    y1 = func(x1)
    y2 = func(x2)
    y3 = func(x3)

    assert y1.shape == get_call_expected_dims(n_samples=17, is_old_api=is_old_api)
    assert y2.shape == get_call_expected_dims(n_samples=func.dim, is_old_api=is_old_api)
    assert y3.shape == get_call_expected_dims(n_samples=1, is_old_api=is_old_api)


def check_func_sample_dims(func, func_log, is_old_api=False):
    x1 = np.random.rand(17, func.dim)
    x2 = np.random.rand(func.dim, func.dim)
    x3 = np.random.rand(1, func.dim)

    x1_sample, y1_sample = func.sample(x1)
    x2_sample, y2_sample = func.sample(x2)
    x3_sample, y3_sample = func.sample(x3)

    np.testing.assert_allclose(x1_sample, np.repeat(x1, repeats=func.n_obs, axis=0))
    np.testing.assert_allclose(x2_sample, np.repeat(x2, repeats=func.n_obs, axis=0))
    np.testing.assert_allclose(x3_sample, np.repeat(x3, repeats=func.n_obs, axis=0))

    assert y1_sample.shape == get_sample_expected_dims(n_samples=17, n_obs=func.n_obs, is_old_api=is_old_api) 
    assert y2_sample.shape == get_sample_expected_dims(n_samples=func.dim, n_obs=func.n_obs, is_old_api=is_old_api)
    assert y3_sample.shape == get_sample_expected_dims(n_samples=1, n_obs=func.n_obs, is_old_api=is_old_api)


def check_func_log_transform(func, func_log, is_old_api=False):
    x1 = np.random.rand(17, func.dim)
    x2 = np.random.rand(func.dim, func.dim)
    x3 = np.random.rand(1, func.dim)

    # Test with log_transform=False
    _call_method = func.f if is_old_api else func.func
    np.testing.assert_allclose(func(x1), _call_method(x1))
    np.testing.assert_allclose(func(x2), _call_method(x2))
    np.testing.assert_allclose(func(x3), _call_method(x3))

    # Test with log_transform=True
    log_eps = func_log.log_eps
    _call_method = func_log.f if is_old_api else func_log.func

    np.testing.assert_allclose(func_log(x1), np.log(_call_method(x1) + log_eps))
    np.testing.assert_allclose(func_log(x2), np.log(_call_method(x2) + log_eps))
    np.testing.assert_allclose(func_log(x3), np.log(_call_method(x3) + log_eps))


def check_func_min(func, atol=1e-5):
    y_min = func(func.glob_min)
    np.testing.assert_allclose(y_min, func.f_min, atol=atol)
