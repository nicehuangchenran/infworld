from infworld.models.dit_model import WanModel
from torchinfo import summary


model=WanModel()
# summary(
#     model,
#     input_size=(1, 3, 224, 224),  # 按你的输入改
#     depth=4,
#     col_names=["input_size", "output_size", "num_params", "trainable"]
# )

print(model)