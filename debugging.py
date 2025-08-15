def print_tensor_info(**tensors):
    """
    Receives multiple tensor variables and prints their shape and sum.

    Example usage:
    print_tensor_info(
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask,
        train_perf_eval=train_perf_eval,
        val_perf_eval=val_perf_eval,
        test_perf_eval=test_perf_eval
    )
    """
    for name, tensor in tensors.items():
        try:
            shape = tuple(tensor.shape)
            total_sum = tensor.sum().item()
            print(f"{name}: shape = {shape}, sum = {total_sum}")
        except AttributeError:
            print(f"{name}: Not a valid tensor-like object")