def count_patients(file_path):
    patient_ids = set()  # 使用集合自动去重
    total_slices = 0

    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip()
            if not line:
                continue  # 跳过空行

            # 提取病人ID（第2到第8个字符，例如"D1516769" → "1516769"）
            if len(line) >= 8:  # 确保行长度足够
                patient_id = line[1:8]  # Python字符串索引从0开始，line[1]是第2个字符
                patient_ids.add(patient_id)
                total_slices += 1

    print(f"总病人数: {len(patient_ids)}")
    print(f"总切片数: {total_slices}")

if __name__ == "__main__":
    file_path = r"D:\UNet_py\dataset_split\dataset_split\test_vol.txt"
    count_patients(file_path)