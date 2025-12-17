import numpy as np


class IntervalEncoder:
    def __init__(self):
        # 8位二进制数范围是0-255
        self.total_numbers = 256
        self.numbers_per_interval = self.total_numbers // 20  # 每个区间12或13个数
        self.remainder = self.total_numbers % 20  # 处理余数

        # 创建两组区间
        self.small_intervals = np.linspace(1e-4, 1e-3, 11)  # 1e-4步长
        self.large_intervals = np.linspace(1e-3, 1e-2, 11)  # 1e-3步长

    def get_interval_boundaries(self, decimal):
        """
        根据十进制数确定其所属区间的边界
        """
        count = 0
        for i in range(20):
            current_count = self.numbers_per_interval + (1 if i < self.remainder else 0)
            if count <= decimal < count + current_count:
                # 确定是在小区间还是大区间
                if i < 10:  # 小区间 (1e-4 ~ 1e-3)
                    return (self.small_intervals[i], self.small_intervals[i + 1], i, count, current_count)
                else:  # 大区间 (1e-3 ~ 1e-2)
                    j = i - 10
                    return (self.large_intervals[j], self.large_intervals[j + 1], i, count, current_count)
            count += current_count
        return None

    def encode(self, binary_str):
        """
        将8位二进制数编码为对应区间内的值
        """
        # 将二进制转换为十进制
        decimal = int(binary_str, 2)

        # 获取区间信息
        interval_info = self.get_interval_boundaries(decimal)
        if interval_info is None:
            raise ValueError("无效的二进制数")

        start, end, interval_idx, count, current_count = interval_info

        # 计算在区间内的相对位置
        relative_pos = (decimal - count) / current_count

        # 线性插值得到最终值
        value = start + relative_pos * (end - start)
        return value, interval_idx

    def decode(self, value):
        """
        将值解码为最接近的8位二进制数
        """
        if value < 1e-4 or value > 1e-2:
            raise ValueError("值超出范围")

        # 确定值所在的区间
        if value < 1e-3:
            intervals = self.small_intervals
            interval_offset = 0
        else:
            intervals = self.large_intervals
            interval_offset = 10

        # 找到具体区间
        for i in range(len(intervals) - 1):
            if intervals[i] <= value < intervals[i + 1]:
                interval_idx = i + interval_offset
                break
        else:
            interval_idx = 19  # 最后一个区间

        # 计算在当前区间内的相对位置
        start = intervals[i]
        end = intervals[i + 1]
        relative_pos = (value - start) / (end - start)

        # 计算对应的整数值
        count = 0
        for j in range(interval_idx):
            count += self.numbers_per_interval + (1 if j < self.remainder else 0)

        current_count = self.numbers_per_interval + (1 if interval_idx < self.remainder else 0)
        decimal = int(count + relative_pos * current_count)

        # 确保值在有效范围内
        decimal = max(0, min(255, decimal))

        return format(decimal, '08b')

def test_encoder():
    encoder = IntervalEncoder()

    # 测试分布均匀性
    interval_counts = np.zeros(20)
    values = []

    print("测试所有可能的8位二进制数的分布:")
    print("-" * 50)

    # 统计每个区间的数量
    for i in range(256):
        binary = format(i, '08b')
        value, interval_idx = encoder.encode(binary)
        interval_counts[interval_idx] += 1
        values.append(value)

    print("每个区间的数量:")
    print("\n1e-4 ~ 1e-3 范围（步长1e-4）:")
    for i in range(10):
        print(f"区间 {i} [{encoder.small_intervals[i]:.4e} - {encoder.small_intervals[i + 1]:.4e}]: "
              f"{int(interval_counts[i])} 个数")

    print("\n1e-3 ~ 1e-2 范围（步长1e-3）:")
    for i in range(10):
        print(f"区间 {i + 10} [{encoder.large_intervals[i]:.3e} - {encoder.large_intervals[i + 1]:.3e}]: "
              f"{int(interval_counts[i + 10])} 个数")

    print(f"\n理论上每个区间应该有: {256 / 20} 个数")

    # 测试随机编解码
    print("\n\n测试随机二进制数的编解码:")
    print("-" * 50)
    for _ in range(5):
        # 生成随机8位二进制数
        random_binary = format(np.random.randint(0, 256), '08b')
        value, interval_idx = encoder.encode(random_binary)
        decoded_binary = encoder.decode(value)
        print(f"原始二进制: {random_binary} -> 值: {value:.8f} (区间 {interval_idx}) -> "
              f"解码二进制: {decoded_binary}")


if __name__ == "__main__":
    test_encoder()
