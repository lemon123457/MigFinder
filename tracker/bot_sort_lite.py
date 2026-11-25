"""
BoT-SORT Lite - 轻量级跟踪器
基于 BoT-SORT 核心算法，兼容 Python 3.10+
包含 Kalman 滤波、IoU 匹配和轻量 bbox 外观匹配，无需 ReID 依赖
"""

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment
from typing import List, Optional, Tuple
from dataclasses import dataclass


@dataclass
class TrackingConfig:
    """跟踪配置"""
    track_high_thresh: float = 0.6      # 高置信度阈值
    track_low_thresh: float = 0.1       # 低置信度阈值
    new_track_thresh: float = 0.7       # 新轨迹阈值
    track_buffer: int = 30              # 丢失轨迹缓冲帧数
    match_thresh: float = 0.8           # 匹配阈值
    proximity_thresh: float = 0.5       # IoU 距离阈值
    confirm_frames: int = 3             # 确认轨迹所需帧数
    use_appearance: bool = True         # 是否使用 bbox 外观特征辅助匹配
    appearance_weight: float = 0.5     # 外观距离在总代价里的权重
    appearance_thresh: float = 0.7      # 外观距离门限，超过后禁止匹配
    appearance_momentum: float = 0.9    # track 外观特征 EMA 更新动量
    appearance_hist_bins: int = 8       # 每个通道的直方图 bin 数


class KalmanFilter:
    """
    简单的 Kalman 滤波器用于边界框跟踪
    状态向量: [cx, cy, aspect_ratio, height, vx, vy, va, vh]
    """

    def __init__(self):
        # 状态转移矩阵
        self.F = np.eye(8)
        self.F[:4, 4:] = np.eye(4)

        # 观测矩阵
        self.H = np.eye(4, 8)

        # 过程噪声协方差
        self.Q = np.eye(8)
        self.Q[4:, 4:] *= 0.01

        # 观测噪声协方差
        self.R = np.eye(4)
        self.R[2:, 2:] *= 10

    def initiate(self, measurement: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """初始化状态"""
        mean = np.zeros(8)
        mean[:4] = measurement

        covariance = np.eye(8)
        covariance[4:, 4:] *= 1000  # 初始速度不确定性大

        return mean, covariance

    def predict(self, mean: np.ndarray, covariance: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """预测下一状态"""
        mean = self.F @ mean
        covariance = self.F @ covariance @ self.F.T + self.Q
        return mean, covariance

    def update(self, mean: np.ndarray, covariance: np.ndarray,
               measurement: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """更新状态"""
        # Kalman 增益
        S = self.H @ covariance @ self.H.T + self.R
        K = covariance @ self.H.T @ np.linalg.inv(S)

        # 更新
        y = measurement - self.H @ mean
        mean = mean + K @ y
        covariance = (np.eye(8) - K @ self.H) @ covariance

        return mean, covariance


class TrackState:
    """轨迹状态"""
    New = 0
    Tracked = 1
    Lost = 2
    Removed = 3


class STrack:
    """单个轨迹"""

    _count = 0
    shared_kalman = KalmanFilter()

    def __init__(self, tlwh: np.ndarray, score: float, class_id: int = 0,
                 feature: Optional[np.ndarray] = None):
        self._tlwh = np.asarray(tlwh, dtype=np.float64)
        self.score = score
        self.class_id = class_id

        self.track_id = 0
        self.is_activated = False
        self.state = TrackState.New

        self.mean = None
        self.covariance = None

        self.frame_id = 0
        self.start_frame = 0
        self.tracklet_len = 0

        self.curr_feat: Optional[np.ndarray] = None
        self.smooth_feat: Optional[np.ndarray] = None
        if feature is not None:
            self.update_features(feature, momentum=0.0)

    @staticmethod
    def _normalize_feature(feature: np.ndarray) -> Optional[np.ndarray]:
        feature = np.asarray(feature, dtype=np.float32).reshape(-1)
        norm = np.linalg.norm(feature)
        if norm <= 1e-12:
            return None
        return feature / norm

    def update_features(self, feature: Optional[np.ndarray], momentum: float = 0.9):
        """使用 EMA 更新当前 track 的 bbox 外观特征。"""
        normalized = self._normalize_feature(feature) if feature is not None else None
        if normalized is None:
            return

        self.curr_feat = normalized
        if self.smooth_feat is None or momentum <= 0:
            self.smooth_feat = normalized
        else:
            self.smooth_feat = momentum * self.smooth_feat + (1.0 - momentum) * normalized
            self.smooth_feat = self._normalize_feature(self.smooth_feat)

    def activate(self, frame_id: int):
        """激活轨迹"""
        STrack._count += 1
        self.track_id = STrack._count

        self.mean, self.covariance = self.shared_kalman.initiate(
            self.tlwh_to_xyah(self._tlwh)
        )

        self.tracklet_len = 0
        self.state = TrackState.Tracked

        if frame_id == 1:
            self.is_activated = True

        self.frame_id = frame_id
        self.start_frame = frame_id

    def re_activate(self, new_track: 'STrack', frame_id: int, new_id: bool = False,
                    appearance_momentum: float = 0.9):
        """重新激活轨迹"""
        self.mean, self.covariance = self.shared_kalman.update(
            self.mean, self.covariance,
            self.tlwh_to_xyah(new_track.tlwh)
        )
        self.tracklet_len = 0
        self.state = TrackState.Tracked
        self.is_activated = True
        self.frame_id = frame_id
        self.score = new_track.score
        self.update_features(new_track.curr_feat, appearance_momentum)

        if new_id:
            STrack._count += 1
            self.track_id = STrack._count

    def update(self, new_track: 'STrack', frame_id: int,
               appearance_momentum: float = 0.9):
        """更新轨迹"""
        self.frame_id = frame_id
        self.tracklet_len += 1

        new_tlwh = new_track.tlwh
        self.mean, self.covariance = self.shared_kalman.update(
            self.mean, self.covariance,
            self.tlwh_to_xyah(new_tlwh)
        )

        self.state = TrackState.Tracked
        self.is_activated = True
        self.score = new_track.score
        self.update_features(new_track.curr_feat, appearance_momentum)

    def predict(self):
        """预测下一状态"""
        if self.mean is not None:
            self.mean, self.covariance = self.shared_kalman.predict(
                self.mean, self.covariance
            )

    def mark_lost(self):
        """标记为丢失"""
        self.state = TrackState.Lost

    def mark_removed(self):
        """标记为移除"""
        self.state = TrackState.Removed

    @staticmethod
    def reset_id():
        """重置 ID 计数器"""
        STrack._count = 0

    @property
    def tlwh(self) -> np.ndarray:
        """获取 top-left-width-height 格式"""
        if self.mean is None:
            return self._tlwh.copy()

        ret = self.mean[:4].copy()
        ret[2] *= ret[3]  # aspect_ratio * height = width
        ret[:2] -= ret[2:] / 2  # center to top-left
        return ret

    @property
    def tlbr(self) -> np.ndarray:
        """获取 top-left-bottom-right 格式"""
        ret = self.tlwh.copy()
        ret[2:] += ret[:2]
        return ret

    @staticmethod
    def tlwh_to_xyah(tlwh: np.ndarray) -> np.ndarray:
        """转换为 center-x, center-y, aspect-ratio, height"""
        ret = np.asarray(tlwh).copy()
        ret[:2] += ret[2:] / 2  # top-left to center
        ret[2] /= ret[3]  # width / height = aspect_ratio
        return ret

    @staticmethod
    def tlbr_to_tlwh(tlbr: np.ndarray) -> np.ndarray:
        """tlbr 转 tlwh"""
        ret = np.asarray(tlbr).copy()
        ret[2:] -= ret[:2]
        return ret

    def __repr__(self):
        return f"STrack(id={self.track_id}, state={self.state})"


def iou_distance(atracks: List[STrack], btracks: List[STrack]) -> np.ndarray:
    """计算两组轨迹之间的 IoU 距离矩阵"""
    if len(atracks) == 0 or len(btracks) == 0:
        return np.zeros((len(atracks), len(btracks)))

    atlbrs = np.array([t.tlbr for t in atracks])
    btlbrs = np.array([t.tlbr for t in btracks])

    ious = np.zeros((len(atlbrs), len(btlbrs)))

    for i, a in enumerate(atlbrs):
        for j, b in enumerate(btlbrs):
            # 计算交集
            x1 = max(a[0], b[0])
            y1 = max(a[1], b[1])
            x2 = min(a[2], b[2])
            y2 = min(a[3], b[3])

            inter = max(0, x2 - x1) * max(0, y2 - y1)

            # 计算并集
            area_a = (a[2] - a[0]) * (a[3] - a[1])
            area_b = (b[2] - b[0]) * (b[3] - b[1])
            union = area_a + area_b - inter

            ious[i, j] = inter / union if union > 0 else 0

    return 1 - ious  # 转换为距离


def appearance_distance(atracks: List[STrack], btracks: List[STrack]) -> np.ndarray:
    """计算两组轨迹/检测之间的外观 cosine distance。"""
    if len(atracks) == 0 or len(btracks) == 0:
        return np.zeros((len(atracks), len(btracks)))

    dists = np.ones((len(atracks), len(btracks)), dtype=np.float32)
    for i, track in enumerate(atracks):
        track_feat = track.smooth_feat
        if track_feat is None:
            continue
        for j, det in enumerate(btracks):
            det_feat = det.curr_feat
            if det_feat is None:
                continue
            similarity = float(np.dot(track_feat, det_feat))
            dists[i, j] = 1.0 - np.clip(similarity, -1.0, 1.0)
    return dists


def linear_assignment(cost_matrix: np.ndarray, thresh: float) -> Tuple[List, List, List]:
    """
    使用匈牙利算法进行线性分配

    Returns:
        matches: 匹配对列表 [(track_idx, det_idx), ...]
        unmatched_a: 未匹配的第一组索引
        unmatched_b: 未匹配的第二组索引
    """
    if cost_matrix.size == 0:
        return [], list(range(cost_matrix.shape[0])), list(range(cost_matrix.shape[1]))

    row_indices, col_indices = linear_sum_assignment(cost_matrix)

    matches = []
    unmatched_a = list(range(cost_matrix.shape[0]))
    unmatched_b = list(range(cost_matrix.shape[1]))

    for row, col in zip(row_indices, col_indices):
        if cost_matrix[row, col] <= thresh:
            matches.append((row, col))
            unmatched_a.remove(row)
            unmatched_b.remove(col)

    return matches, unmatched_a, unmatched_b


class BoTSORTLite:
    """
    BoT-SORT Lite 跟踪器
    基于 Kalman 滤波、IoU 和 bbox 外观特征的多目标跟踪
    """

    def __init__(self, config: Optional[TrackingConfig] = None, frame_rate: int = 30):
        self.config = config or TrackingConfig()
        self.frame_rate = frame_rate

        self.tracked_stracks: List[STrack] = []
        self.lost_stracks: List[STrack] = []
        self.removed_stracks: List[STrack] = []

        self.frame_id = 0
        self.max_time_lost = int(frame_rate / 30.0 * self.config.track_buffer)

        STrack.reset_id()

    @staticmethod
    def _joint_stracks(tlista: List[STrack], tlistb: List[STrack]) -> List[STrack]:
        """按 track_id 合并轨迹列表，保持稳定顺序。"""
        exists = {}
        res = []
        for t in tlista:
            exists[t.track_id] = True
            res.append(t)
        for t in tlistb:
            if not exists.get(t.track_id, False):
                exists[t.track_id] = True
                res.append(t)
        return res

    @staticmethod
    def _sub_stracks(tlista: List[STrack], tlistb: List[STrack]) -> List[STrack]:
        """从 tlista 中移除在 tlistb 里出现过的 track_id。"""
        stracks = {t.track_id: t for t in tlista}
        for t in tlistb:
            if t.track_id in stracks:
                del stracks[t.track_id]
        return list(stracks.values())

    @staticmethod
    def _extract_bbox_feature(img: Optional[np.ndarray], tlbr: np.ndarray,
                              hist_bins: int = 8) -> Optional[np.ndarray]:
        """从 bbox crop 中提取轻量外观特征：颜色统计、直方图和低分辨率纹理。"""
        if img is None:
            return None

        h, w = img.shape[:2]
        x1, y1, x2, y2 = tlbr
        x1 = int(np.clip(np.floor(x1), 0, max(w - 1, 0)))
        y1 = int(np.clip(np.floor(y1), 0, max(h - 1, 0)))
        x2 = int(np.clip(np.ceil(x2), 0, w))
        y2 = int(np.clip(np.ceil(y2), 0, h))
        if x2 <= x1 or y2 <= y1:
            return None

        crop = img[y1:y2, x1:x2]
        if crop.size == 0:
            return None
        if crop.ndim == 2:
            crop = np.repeat(crop[:, :, None], 3, axis=2)
        elif crop.shape[2] > 3:
            crop = crop[:, :, :3]

        crop = crop.astype(np.float32) / 255.0
        mean = crop.reshape(-1, crop.shape[2]).mean(axis=0)
        std = crop.reshape(-1, crop.shape[2]).std(axis=0)

        hists = []
        for channel in range(crop.shape[2]):
            hist, _ = np.histogram(crop[:, :, channel], bins=hist_bins, range=(0.0, 1.0))
            hist = hist.astype(np.float32)
            hist_sum = hist.sum()
            if hist_sum > 0:
                hist /= hist_sum
            hists.append(hist)

        gray = crop.mean(axis=2)
        texture = cv2.resize(gray, (8, 8), interpolation=cv2.INTER_AREA).reshape(-1)

        feature = np.concatenate([mean, std, *hists, texture]).astype(np.float32)
        norm = np.linalg.norm(feature)
        if norm <= 1e-12:
            return None
        return feature / norm

    def _extract_detection_features(self, detections: np.ndarray,
                                    img: Optional[np.ndarray]) -> List[Optional[np.ndarray]]:
        if not self.config.use_appearance or img is None:
            return [None] * len(detections)
        return [
            self._extract_bbox_feature(img, det[:4], self.config.appearance_hist_bins)
            for det in detections
        ]

    def _matching_distance(self, atracks: List[STrack], btracks: List[STrack]) -> np.ndarray:
        """组合 IoU distance 和 bbox 外观 distance，返回匈牙利匹配代价矩阵。"""
        iou_dists = iou_distance(atracks, btracks)
        if (
            not self.config.use_appearance
            or self.config.appearance_weight <= 0
            or len(atracks) == 0
            or len(btracks) == 0
        ):
            return iou_dists

        app_dists = appearance_distance(atracks, btracks)
        weight = float(np.clip(self.config.appearance_weight, 0.0, 1.0))
        combined = (1.0 - weight) * iou_dists + weight * app_dists
        valid_app = np.array(
            [
                [track.smooth_feat is not None and det.curr_feat is not None for det in btracks]
                for track in atracks
            ],
            dtype=bool
        )
        combined[valid_app & (app_dists > self.config.appearance_thresh)] = 1e6
        return combined

    def update(self, detections: np.ndarray, img: Optional[np.ndarray] = None) -> List[STrack]:
        """
        更新跟踪器

        Args:
            detections: (N, 5) 数组 [x1, y1, x2, y2, score] 或
                       (N, 6) 数组 [x1, y1, x2, y2, score, class_id]
            img: 当前帧图像 (可选，用于提取 bbox 外观特征)

        Returns:
            当前帧的活跃轨迹列表
        """
        self.frame_id += 1

        activated_stracks = []
        refind_stracks = []
        lost_stracks = []
        removed_stracks = []

        # 处理检测结果
        if len(detections) == 0:
            detections = np.empty((0, 5))

        # 分离高/低置信度检测
        scores = detections[:, 4]
        det_features = self._extract_detection_features(detections, img)

        high_inds = scores >= self.config.track_high_thresh
        low_inds = (scores >= self.config.track_low_thresh) & (scores < self.config.track_high_thresh)

        dets_high = detections[high_inds]
        dets_low = detections[low_inds]
        features_high = [det_features[i] for i in np.flatnonzero(high_inds)]
        features_low = [det_features[i] for i in np.flatnonzero(low_inds)]

        # 创建检测 STrack
        detections_high = [
            STrack(STrack.tlbr_to_tlwh(det[:4]), det[4],
                   int(det[5]) if len(det) > 5 else 0, feature)
            for det, feature in zip(dets_high, features_high)
        ]
        detections_low = [
            STrack(STrack.tlbr_to_tlwh(det[:4]), det[4],
                   int(det[5]) if len(det) > 5 else 0, feature)
            for det, feature in zip(dets_low, features_low)
        ]

        # 分离已激活和未激活轨迹
        unconfirmed = [t for t in self.tracked_stracks if not t.is_activated]
        tracked = [t for t in self.tracked_stracks if t.is_activated]

        # 合并跟踪和丢失轨迹
        strack_pool = tracked + self.lost_stracks

        # 预测
        for t in strack_pool:
            t.predict()

        # === 第一次关联：高置信度检测 ===
        dists = self._matching_distance(strack_pool, detections_high)
        matches, u_track, u_detection = linear_assignment(
            dists, self.config.match_thresh
        )

        for itracked, idet in matches:
            track = strack_pool[itracked]
            det = detections_high[idet]
            if track.state == TrackState.Tracked:
                track.update(det, self.frame_id, self.config.appearance_momentum)
                activated_stracks.append(track)
            else:
                track.re_activate(det, self.frame_id,
                                  appearance_momentum=self.config.appearance_momentum)
                refind_stracks.append(track)

        # === 第二次关联：低置信度检测 ===
        r_tracked = [strack_pool[i] for i in u_track if strack_pool[i].state == TrackState.Tracked]

        dists = self._matching_distance(r_tracked, detections_low)
        matches, u_track_2, u_detection_low = linear_assignment(
            dists, 1 - 0.5  # 更宽松的阈值
        )

        for itracked, idet in matches:
            track = r_tracked[itracked]
            det = detections_low[idet]
            track.update(det, self.frame_id, self.config.appearance_momentum)
            activated_stracks.append(track)

        # 未匹配轨迹标记为丢失
        for i in u_track_2:
            track = r_tracked[i]
            if track.state != TrackState.Lost:
                track.mark_lost()
                lost_stracks.append(track)

        # === 处理未确认轨迹 ===
        detections_remain = [detections_high[i] for i in u_detection]

        dists = self._matching_distance(unconfirmed, detections_remain)
        matches, u_unconfirmed, u_detection_2 = linear_assignment(
            dists, 1 - 0.7
        )

        for itracked, idet in matches:
            unconfirmed[itracked].update(
                detections_remain[idet],
                self.frame_id,
                self.config.appearance_momentum
            )
            activated_stracks.append(unconfirmed[itracked])

        for i in u_unconfirmed:
            track = unconfirmed[i]
            track.mark_removed()
            removed_stracks.append(track)

        # === 初始化新轨迹 ===
        for i in u_detection_2:
            track = detections_remain[i]
            if track.score >= self.config.new_track_thresh:
                track.activate(self.frame_id)
                activated_stracks.append(track)

        # === 更新轨迹状态 ===
        for track in self.lost_stracks:
            if self.frame_id - track.frame_id > self.max_time_lost:
                track.mark_removed()
                removed_stracks.append(track)

        tracked_active = [t for t in self.tracked_stracks if t.state == TrackState.Tracked]
        self.tracked_stracks = self._joint_stracks(tracked_active, activated_stracks)
        self.tracked_stracks = self._joint_stracks(self.tracked_stracks, refind_stracks)

        # 清理丢失轨迹
        self.lost_stracks = self._sub_stracks(self.lost_stracks, self.tracked_stracks)
        self.lost_stracks.extend(lost_stracks)
        self.lost_stracks = self._sub_stracks(self.lost_stracks, self.removed_stracks)
        self.removed_stracks.extend(removed_stracks)
        self.lost_stracks = [t for t in self.lost_stracks if t.state == TrackState.Lost]

        # 返回激活的轨迹
        output = [t for t in self.tracked_stracks if t.is_activated]
        return output

    def reset(self):
        """重置跟踪器"""
        self.tracked_stracks = []
        self.lost_stracks = []
        self.removed_stracks = []
        self.frame_id = 0
        STrack.reset_id()
