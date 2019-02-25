"""
Config
---------
Set device to 'cpu' or 'cuda:index'
Default: cuda:0 if available, otherwise cpu

"""
import torch

user_device = None







default_device = 'cpu'
# default_device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
device_ = default_device if user_device is None else user_device
device = torch.device(device_)







#
# def trackcalls(func):
#     @functools.wraps(func)
#     def wrapper(*args, **kwargs):
#         wrapper.has_been_called = True
#         return func(*args, **kwargs)
#     wrapper.has_been_called = False
#     return wrapper
#
# @trackcalls
# def set_device(device_):
#     global device
#     if isinstance(device_, str):
#         if 'cuda' in device_:
#             assert torch.cuda.is_available(), "Cuda requested but not available"
#             device = torch.device(device_)
#         elif device_ == 'cpu':
#             device = torch.device(device_)
#         else:
#             raise ValueError("Device is not recongnized")
#     elif isinstance(device_, int):
#         device = torch.device(device_)
#
# device = None
