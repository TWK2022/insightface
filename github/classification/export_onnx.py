import os
import torch
import argparse
import pandas as pd

# -------------------------------------------------------------------------------------------------------------------- #
# 设置
parser = argparse.ArgumentParser(description='将pt模型字典中的模型转为onnx，同时导出类别信息')
parser.add_argument('--weight', default='best.pt', type=str, help='|模型位置|')
parser.add_argument('--input_size', default=160, type=int, help='|输入图片大小|')
parser.add_argument('--batch', default=0, type=int, help='|输入图片批量，0为动态|')
parser.add_argument('--sim', default=True, type=bool, help='|使用onnxsim压缩简化模型|')
parser.add_argument('--float16', default=False, type=bool, help='|转换的onnx模型数据类型，float16需要GPU，False时为float32|')
args = parser.parse_args()
args.weight = args.weight.split('.')[0] + '.pt'
args.save_name = args.weight.split('.')[0] + '.onnx'
args.device = 'cuda' if args.float16 else 'cpu'
# -------------------------------------------------------------------------------------------------------------------- #
# 初步检查
assert os.path.exists(args.weight), '没有找到模型{}'.format(args.weight)
if args.float16:
    assert torch.cuda.is_available(), 'cuda不可用，因此无法转为float16'


# -------------------------------------------------------------------------------------------------------------------- #
# 程序
def export_onnx():
    model_dict = torch.load(args.weight, map_location='cpu')
    model = model_dict['model']
    model.half().eval().to(args.device) if args.float16 else model.float().eval().to(args.device)
    input_shape = torch.zeros(1, 3, args.input_size, args.input_size,
                              dtype=torch.float16 if args.float16 else torch.float32).to(args.device)
    torch.onnx.export(model, input_shape, args.save_name,
                      opset_version=12, input_names=['input'], output_names=['output'],
                      dynamic_axes={'input': {args.batch: 'batch_size'}, 'output': {args.batch: 'batch_size'}})
    print('| 转为onnx模型成功:{} |'.format(args.save_name))
    cls = model_dict['class']
    cls_df = pd.DataFrame(cls, columns=['class'])
    cls_df.to_csv('class.csv', index=False, header=False)
    if args.sim:
        import onnx
        import onnxsim

        model_onnx = onnx.load(args.save_name)
        model_simplify, check = onnxsim.simplify(model_onnx)
        onnx.save(model_simplify, args.save_name)
        print('| 使用onnxsim简化模型成功 |')


if __name__ == '__main__':
    export_onnx()