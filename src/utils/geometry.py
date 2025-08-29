# src/utils/geometry.py
import numpy as np

def rotation_matrix(yaw: float) -> np.ndarray:
    """
    Creates a 2D rotation matrix from a yaw angle.

    Args:
        yaw: The yaw angle in radians.

    Returns:
        A 2x2 NumPy array representing the rotation matrix.
    """
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, -s], [s, c]])

def transform_points(points: np.ndarray, pose: np.ndarray) -> np.ndarray:
    """
    Transforms a set of points from a global frame to a local frame defined by a pose.

    Args:
        points: A NumPy array of shape (N, 2) or (N, 3) representing the points to transform.
        pose: A NumPy array of shape (3,) representing the [x, y, yaw] of the local frame's origin.

    Returns:
        A NumPy array of the same shape as `points`, with the points in the local frame.
    """
    ego_x, ego_y, ego_yaw = pose
    
    # Create the inverse transformation matrix
    # First, translate the points to the ego vehicle's origin
    translated_points = points[:, :2] - np.array([ego_x, ego_y])
    
    # Then, rotate them by the inverse of the ego vehicle's yaw
    # The inverse of a rotation matrix is its transpose
    inv_rot_matrix = rotation_matrix(-ego_yaw)
    
    # Apply the rotation
    transformed_points_2d = translated_points @ inv_rot_matrix.T
    
    if points.shape[1] == 3:
        # If the points have a z-coordinate, re-add it after transformation
        # Assuming z is relative to the ground, so it doesn't change
        return np.hstack([transformed_points_2d, points[:, 2:3]])
    
    return transformed_points_2d

def resample_polyline(polyline: np.ndarray, num_points: int) -> np.ndarray:
    """
    (Definitive, Robust Version) Resamples a polyline to a fixed number of points.
    Handles zero-length and single-point polylines gracefully.

    Args:
        polyline: A NumPy array of shape (M, D), where D is the dimension (e.g., 2 or 3).
        num_points: The desired number of points in the output polyline.

    Returns:
        A NumPy array of shape (num_points, D).
    """
    num_original_points, num_dims = polyline.shape

    # If the polyline is empty, return zeros of the correct shape.
    if num_original_points == 0:
        return np.zeros((num_points, num_dims), dtype=polyline.dtype)

    # If the polyline has only one point, or all points are identical,
    # simply repeat the first point.
    if num_original_points < 2 or np.all(polyline == polyline[0]):
        return np.tile(polyline[0], (num_points, 1))
        
    # Calculate the cumulative distance along the polyline
    distances = np.sqrt(np.sum(np.diff(polyline, axis=0)**2, axis=1))
    cumulative_distances = np.insert(np.cumsum(distances), 0, 0)
    
    total_distance = cumulative_distances[-1]

    # If total distance is negligible, it's effectively a single point.
    if total_distance < 1e-4:
        return np.tile(polyline[0], (num_points, 1))

    # Create a new set of evenly spaced distances for resampling
    new_distances = np.linspace(0, total_distance, num_points)
    
    # Interpolate each dimension (x, y, z, etc.)
    resampled = np.zeros((num_points, num_dims), dtype=polyline.dtype)
    for i in range(num_dims):
        resampled[:, i] = np.interp(new_distances, cumulative_distances, polyline[:, i])
    
    # Final check for any NaNs/infs that might have been created
    if not np.all(np.isfinite(resampled)):
        # Fallback in case of an unexpected numerical issue
        return np.tile(polyline[0], (num_points, 1))
    
    return resampled

def perpendicular_distance_point_to_line_segment(point: np.ndarray, line_start: np.ndarray, line_end: np.ndarray) -> float:
    """
    Calculates the perpendicular distance from a point to a line segment.

    Args:
        point: A NumPy array of shape (2,) for the point [x, y].
        line_start: A NumPy array of shape (2,) for the start of the segment.
        line_end: A NumPy array of shape (2,) for the end of the segment.

    Returns:
        The minimum distance from the point to the line segment.
    """
    # Vector from line_start to line_end
    line_vec = line_end - line_start
    # Vector from line_start to the point
    point_vec = point - line_start
    
    line_len_sq = np.dot(line_vec, line_vec)
    if line_len_sq == 0.0:
        return np.linalg.norm(point_vec) # Line is a point

    # Project point_vec onto line_vec
    # t is the projection factor. If 0 <= t <= 1, the projection is on the segment.
    t = max(0, min(1, np.dot(point_vec, line_vec) / line_len_sq))
    
    # The closest point on the line segment to the given point
    closest_point_on_segment = line_start + t * line_vec
    
    # Return the distance between the point and its projection
    return np.linalg.norm(point - closest_point_on_segment)