from tree_height.data.dataset import TFRecordDataset
import torch
from torch.utils.data import DataLoader
import glob, os


def combine_norm_args(norm_args) -> dict:

    def update(a_samples: int, a_mean: torch.Tensor, a_m2: torch.Tensor,
               b_samples_: int, b_mean_: torch.Tensor, b_m2_: torch.Tensor):
        all_samples = a_samples + b_samples_
        delta = b_mean_ - a_mean
        all_mean = delta * b_samples_ / all_samples + a_mean
        all_m2 = delta ** 2 * a_samples * b_samples_ / all_samples + a_m2 + b_m2_

        return all_samples, all_mean, all_m2

    samples = 0
    mean = torch.zeros(1)
    std = torch.zeros(1)
    m2 = torch.zeros(1)

    for args in norm_args:
        b_mean, b_std, b_samples = args["mean"], args["std"], args["samples"]
        b_m2 = torch.tensor(b_std) ** 2 * b_samples
        samples, mean, m2 = update(samples, mean, m2, b_samples, torch.tensor(b_mean), b_m2)

    if samples > 0:
        mean = mean.tolist()
        std = torch.sqrt(m2 / samples).tolist()

    return {"mean": mean, "std": std, "samples": samples}


def nanstd(o, dim, keepdim=False):
    #https://medium.com/@allenyllee/pytorch%E7%AD%86%E8%A8%98-%E5%A6%82%E4%BD%95%E8%A8%88%E7%AE%97%E9%9D%9E0%E9%A0%85%E7%9A%84mean%E8%B7%9Fstd-791e4a876245
    result = torch.sqrt(
        torch.nanmean(
            torch.pow(torch.abs(o - torch.nanmean(o, dim=dim).unsqueeze(dim)), 2),
            dim=dim
        )
    )

    if keepdim:
        result = result.unsqueeze(dim)

    return result


def main():
    trainTFRecordPath = glob.glob(os.path.join('/mnt/code/deep_learning/tree_height/dataset/train', '*.tfrecord.gz'))
    trainTFRecordDataset = TFRecordDataset(trainTFRecordPath, is_predict=False)

    samples = 0
    x_mean_sum = torch.zeros(1)
    x_square_mean_sum = torch.zeros(1)

    y_mean_sum = torch.zeros(1)
    y_square_mean_sum = torch.zeros(1)

    y_args = []

    for x, y in DataLoader(trainTFRecordDataset, batch_size=20):
        # 计算 x 的平均值，标准差
        samples += x.shape[0]
        x_mean_sum = torch.sum(torch.mean(x, dim=(2, 3)), dim=0) + x_mean_sum
        x_square_mean_sum = torch.sum(torch.mean(torch.square(x), dim=(2, 3)), dim=0) + x_square_mean_sum

        # 计算 y 的平均值，标准差
        # 需要去掉等于 0 的像素，比较麻烦
        non_y = y[y != 0]
        non_nan_count = len(non_y)
        mean = torch.mean(non_y)
        std = torch.std(non_y)

        y_args.append({'mean': mean.tolist(), 'std': std.tolist(), 'samples': non_nan_count})

    x_mean = (x_mean_sum / samples)
    x_std = torch.sqrt(x_square_mean_sum / samples - torch.square(x_mean))

    print({'x_mean': x_mean.tolist(), 'x_std': x_std.tolist(), 'samples': samples})
    print(combine_norm_args(y_args))

if __name__ == '__main__':
    main()
