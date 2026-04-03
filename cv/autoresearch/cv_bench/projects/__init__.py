from __future__ import annotations

from cv_bench.projects.project_climate_patch_segmentation import run as run_climate_patch_segmentation
from cv_bench.projects.project_constrained_depth_estimation import run as run_constrained_depth_estimation
from cv_bench.projects.project_document_layout_analysis import run as run_document_layout_analysis
from cv_bench.projects.project_edge_semantic_segmentation import run as run_edge_semantic_segmentation
from cv_bench.projects.project_event_camera import run as run_event_camera
from cv_bench.projects.project_gaze_tracking import run as run_gaze_tracking
from cv_bench.projects.project_hybrid_edge_tracking import run as run_hybrid_edge_tracking
from cv_bench.projects.project_hyperspectral_agriculture import run as run_hyperspectral_agriculture
from cv_bench.projects.project_medical_ultrasound_segmentation import run as run_medical_ultrasound_segmentation
from cv_bench.projects.project_micro_expression_recognition import run as run_micro_expression_recognition
from cv_bench.projects.project_multicamera_calibration import run as run_multicamera_calibration
from cv_bench.projects.project_procedural_edge_case_generation import run as run_procedural_edge_case_generation
from cv_bench.projects.project_sensor_degraded_tracking import run as run_sensor_degraded_tracking
from cv_bench.projects.project_synthetic_graphics_validation import run as run_synthetic_graphics_validation
from cv_bench.projects.project_visual_anomaly_patchcore import run as run_visual_anomaly_patchcore

__all__ = [
	"run_climate_patch_segmentation",
	"run_constrained_depth_estimation",
	"run_document_layout_analysis",
	"run_edge_semantic_segmentation",
	"run_event_camera",
	"run_gaze_tracking",
	"run_hybrid_edge_tracking",
	"run_hyperspectral_agriculture",
	"run_medical_ultrasound_segmentation",
	"run_micro_expression_recognition",
	"run_multicamera_calibration",
	"run_procedural_edge_case_generation",
	"run_sensor_degraded_tracking",
	"run_synthetic_graphics_validation",
	"run_visual_anomaly_patchcore",
]