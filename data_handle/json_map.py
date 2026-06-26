import json
import os

# ===== 自定义标签映射表 =====
# 格式：'旧标签': '新标签'
# 例如：将 'water' 改为 'inland-water'
LABEL_MAP = {
    "water":"inland-water",
    "paved-area":"concrete-pavement",
    "vegetation":"tall-vegetation",
    "grass":"low-vegetation",
    "utility-pole":"concrete-pole",
    "tree":"woods"
}


# ===== 脚本主体 =====

def convert_labels_in_json(input_path, output_path=None, label_map=None):
    """
    读取 JSON 文件，将 shapes 中的 label 按映射表替换，并保存。

    Args:
        input_path (str): 输入 JSON 文件路径
        output_path (str): 输出 JSON 文件路径，若为 None 则覆盖原文件
        label_map (dict): 标签映射字典，若为 None 则使用全局 LABEL_MAP
    """
    if label_map is None:
        label_map = LABEL_MAP

    if not label_map:
        print("警告：映射表为空，没有标签会被修改。")

    # 读取 JSON
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 统计修改次数
    modified_count = 0

    # 遍历所有 shapes
    if 'shapes' in data and isinstance(data['shapes'], list):
        for shape in data['shapes']:
            if 'label' in shape:
                old_label = shape['label']
                if old_label in label_map:
                    new_label = label_map[old_label]
                    shape['label'] = new_label
                    modified_count += 1
                    print(f"修改标签: '{old_label}' -> '{new_label}'")

    # 确定输出路径
    if output_path is None:
        output_path = input_path  # 覆盖原文件

    # 保存 JSON（保留原始格式，缩进保持与原文件一致，这里使用 2 空格）
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"\n完成！共修改 {modified_count} 个标签。")
    print(f"结果已保存至: {output_path}")


# ===== 使用示例 =====
if __name__ == "__main__":
    import os

    json_dir=r"E:\add_road_test\jsons"
    for file_name in os.listdir(json_dir):
        json_path=os.path.join(json_dir,file_name)
        convert_labels_in_json(json_path, json_path)