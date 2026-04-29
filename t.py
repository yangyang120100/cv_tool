import numpy as np


class DetectionEvaluator:
    def __init__(self, iou_threshold=0.5):
        self.iou_threshold = iou_threshold

    def calculate_iou(self, box1, box2):
        """
        计算两个水平框的 IoU
        box 格式: [x1, y1, x2, y2]
        """
        x_left = max(box1[0], box2[0])
        y_top = max(box1[1], box2[1])
        x_right = min(box1[2], box2[2])
        y_bottom = min(box1[3], box2[3])

        if x_right < x_left or y_bottom < y_top:
            return 0.0

        intersection_area = (x_right - x_left) * (y_bottom - y_top)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])

        union_area = area1 + area2 - intersection_area
        return intersection_area / union_area if union_area > 0 else 0

    def evaluate(self, all_predictions, all_ground_truths):
        """
        all_predictions: 列表，每个元素为 [img_id, x1, y1, x2, y2, score, class_id]
        all_ground_truths: 字典，格式为 {img_id: [[x1, y1, x2, y2, class_id], ...]}
        """
        # 1. 按照置信度从高到低排序
        all_predictions.sort(key=lambda x: x[5], reverse=True)

        num_preds = len(all_predictions)
        tp = np.zeros(num_preds)
        fp = np.zeros(num_preds)

        # 记录每个图像中 GT 的匹配状态
        # gt_matched = {img_id: [False, False, ...]}
        gt_matched = {
            img_id: [False] * len(gts)
            for img_id, gts in all_ground_truths.items()
        }

        # 统计总的 GT 数量（用于计算 Recall）
        total_gts = sum(len(gts) for gts in all_ground_truths.values())

        # 2. 贪婪匹配逻辑
        for i, pred in enumerate(all_predictions):
            img_id, px1, py1, px2, py2, score, pcls = pred
            pred_box = [px1, py1, px2, py2]

            best_iou = -1
            best_gt_idx = -1

            if img_id in all_ground_truths:
                gts = all_ground_truths[img_id]
                for gt_idx, gt in enumerate(gts):
                    gx1, gy1, gx2, gy2, gcls = gt

                    # 类别必须一致
                    if pcls != gcls:
                        continue

                    iou = self.calculate_iou(pred_box, [gx1, gy1, gx2, gy2])
                    if iou > best_iou:
                        best_iou = iou
                        best_gt_idx = gt_idx

            # 3. 判定 TP 或 FP
            if best_iou >= self.iou_threshold:
                if not gt_matched[img_id][best_gt_idx]:
                    tp[i] = 1
                    gt_matched[img_id][best_gt_idx] = True
                else:
                    # 该 GT 已被更高置信度的预测框占用
                    fp[i] = 1
            else:
                # IoU 不足或未找到对应类别的 GT
                fp[i] = 1

        # 4. 计算指标
        tp_cumsum = np.cumsum(tp)
        fp_cumsum = np.cumsum(fp)
        precisions = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-10)
        recalls = tp_cumsum / (total_gts + 1e-10)

        return precisions, recalls


# --- 使用示例 ---
if __name__ == "__main__":
    # 模拟数据：[img_id, x1, y1, x2, y2, confidence, class_id]
    mock_preds = [
        [1, 10, 10, 50, 50, 0.95, 0],
        [1, 12, 12, 52, 52, 0.88, 0],  # 这个应该是 FP，因为和上面重复匹配同一个 GT
        [2, 20, 20, 60, 60, 0.92, 0]
    ]

    # 真实标签：{img_id: [[x1, y1, x2, y2, class_id], ...]}
    mock_gts = {
        1: [[10, 10, 50, 50, 0]],
        2: [[20, 20, 60, 60, 0]]
    }

    evaluator = DetectionEvaluator(iou_threshold=0.5)
    p, r = evaluator.evaluate(mock_preds, mock_gts)

    print(f"最高 Recall: {r[-1]:.2f}")
    print(f"对应 Precision: {p[-1]:.2f}")