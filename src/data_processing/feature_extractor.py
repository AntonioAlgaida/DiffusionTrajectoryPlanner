# src/data_processing/feature_extractor.py


import numpy as np
from typing import Dict, Union, List, Tuple
from scipy.spatial import cKDTree
from src.utils import geometry
from numpy.lib.npyio import NpzFile


class FeatureExtractor:
    """
    (V4 - Diffusion Policy Version)
    
    A robust feature extractor that converts raw scenario data from .npz files
    into a structured dictionary of feature tensors. This version is specifically
    designed for the Diffusion Policy project. It includes:
    - Rich SDC history (ego_history) instead of a single state.
    - Correctly structured, non-flattened map geometry.
    - Goal-conditioning for long-horizon planning.
    """
    def __init__(self, config: Dict):
        self.features_config = config['features']
        self.relevance_config = config['features']['relevance']
        # self.map_config = config.get('map', {})
        print("FeatureExtractor (V4 - Diffusion Policy) initialized.")
        
    def extract_features(
        self, 
        source: Union[Dict[str, np.ndarray], NpzFile],
        timestep_index: int
    ) -> Dict[str, np.ndarray]:
        """Main public method. Orchestrates the feature extraction pipeline."""
        is_parked_mask_full = self._classify_parked_vehicles(source)

        try:
            (sdc_state_global, other_agents_global, other_agent_types, original_indices,
             lane_polylines_global, traffic_lights_global, sdc_route, sdc_valid_mask) = self._unpack_npz_data(source, timestep_index)
            
        except ValueError as e:
            raise ValueError(f"Failed to unpack data at timestep {timestep_index}: {e}")

        ego_pose = sdc_state_global[[0, 1, 6]] # [global_x, global_y, global_yaw]
        
        other_agents_ego = self._transform_agents_to_ego(other_agents_global, ego_pose)
        lane_polylines_ego = [geometry.transform_points(p, ego_pose) for p in lane_polylines_global]
        
        # --- MODIFIED CALLS ---
        ego_history, ego_history_mask = self._get_ego_history_features(sdc_route, sdc_valid_mask, timestep_index, ego_pose)
        agent_features, agents_mask = self._get_agent_features(
            other_agents_ego, other_agent_types, original_indices, is_parked_mask_full
        )
        map_features, map_mask = self._get_map_features(lane_polylines_ego) # Now returns (64, 10, 2)
        traffic_light_features, tl_mask = self._get_traffic_light_features(traffic_lights_global, ego_pose)
        goal_features, goal_mask = self._get_goal_features(sdc_route, timestep_index, ego_pose)

        # --- MODIFIED DICTIONARY ---
        feature_dict = {
            'ego_history': ego_history, 'ego_history_mask': ego_history_mask,
            'agents': agent_features, 'agents_mask': agents_mask,
            'map': map_features, 'map_mask': map_mask,
            'traffic_lights': traffic_light_features, 'traffic_lights_mask': tl_mask,
            'goal': goal_features, 'goal_mask': goal_mask,
        }
        
        for key, tensor in feature_dict.items():
            if not np.all(np.isfinite(tensor)):
                raise ValueError(f"Invalid number (NaN/inf) in feature '{key}' at timestep {timestep_index}.")
        
        return feature_dict

    def _unpack_npz_data(self, data: Dict, timestep: int) -> Tuple:
        """Unpacks and filters all necessary raw data arrays from the .npz file."""
        sdc_track_index = data['sdc_track_index'].item() # Use .item() for safety
        
        if not data['valid_mask'][sdc_track_index, timestep]:
            raise ValueError(f"SDC not valid at timestep {timestep}")

        states_at_t = data['all_agent_trajectories'][:, timestep, :]
        valid_mask_at_t = data['valid_mask'][:, timestep]
        sdc_state_global = states_at_t[sdc_track_index]
        
        # Select all agents that are valid AND are not the SDC
        other_agents_mask = valid_mask_at_t & (np.arange(len(valid_mask_at_t)) != sdc_track_index)
        original_indices = np.where(other_agents_mask)[0]

        other_agents_global = states_at_t[other_agents_mask]
        other_agent_types = data['object_types'][other_agents_mask]
        
        # Select map features that are lane centerlines
        map_polylines = list(data['map_polylines'])
        map_types = list(data['map_polyline_types'])
        
        # Include TYPE_UNDEFINED (ID 0) as it often represents lanes in intersections.
        lane_polylines_global = [p for p, t in zip(map_polylines, map_types) if t in {0, 1, 2, 3}]
        
        # Select traffic light states that are valid
        dynamic_map_states_t = data['dynamic_map_states'][timestep, :, :]
        traffic_lights_global = dynamic_map_states_t[dynamic_map_states_t[:, 0] > 0]
        
        sdc_route = data['sdc_route']
        sdc_valid_mask = data['valid_mask'][sdc_track_index, :]
        
        return sdc_state_global, other_agents_global, other_agent_types, original_indices, lane_polylines_global, traffic_lights_global, sdc_route, sdc_valid_mask

    def _transform_agents_to_ego(self, agents_global: np.ndarray, ego_pose: np.ndarray) -> np.ndarray:
        """Transforms agent positions, velocities, and headings to be ego-centric."""
        if agents_global.shape[0] == 0:
            return agents_global
        
        agents_ego = agents_global.copy()
        # Transform positions
        agents_ego[:, :2] = geometry.transform_points(agents_global[:, :2], ego_pose)
        # Transform velocities
        rot_mat = geometry.rotation_matrix(-ego_pose[2])
        agents_ego[:, 7:9] = agents_global[:, 7:9] @ rot_mat.T
        # Transform headings and re-normalize to [-pi, pi]
        agents_ego[:, 6] = (agents_global[:, 6] - ego_pose[2] + np.pi) % (2 * np.pi) - np.pi
        return agents_ego

    def _get_ego_history_features(self, sdc_route: np.ndarray, sdc_valid_mask: np.ndarray, current_timestep: int, ego_pose: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        --- NEW METHOD ---
        Extracts the SDC's recent 1-second trajectory history.
        """
        history_steps = 11 # t-10 to t=0, so 11 steps
        
        # Initialize with zeros
        ego_history = np.zeros((history_steps, 3), dtype=np.float32) # x, y, heading
        ego_history_mask = np.zeros(history_steps, dtype=bool)

        start_idx = max(0, current_timestep - history_steps + 1)
        end_idx = current_timestep + 1
        
        # Get the global history and its validity
        history_global = sdc_route[start_idx:end_idx]
        valid_history_mask = sdc_valid_mask[start_idx:end_idx]

        # Transform valid history points to the current ego-centric frame
        if np.any(valid_history_mask):
            positions_ego = geometry.transform_points(history_global[:, :2], ego_pose)
            headings_relative = (history_global[:, 6] - ego_pose[2] + np.pi) % (2 * np.pi) - np.pi
            
            # Place the valid, transformed history at the END of the array
            # This ensures padding is at the beginning if history is short
            offset = history_steps - len(history_global)
            ego_history[offset:, 0:2] = positions_ego
            ego_history[offset:, 2] = headings_relative
            ego_history_mask[offset:] = valid_history_mask

        return ego_history, ego_history_mask
    
    def _get_ego_features(self, sdc_state_global: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Calculates the SDC's own kinematic features."""
        # Feature: [speed]
        speed = np.linalg.norm(sdc_state_global[7:9])
        # Return a mask for consistency, although it's always True for the ego.
        return np.array([speed], dtype=np.float32), np.array([True], dtype=bool)

    def _get_agent_features(
            self, other_agents_ego: np.ndarray, other_agent_types: np.ndarray, 
            original_indices: np.ndarray, is_parked_mask_full: np.ndarray
        ) -> Tuple[np.ndarray, np.ndarray]:
        """
        --- UPGRADED METHOD ---
        Extracts features for the N most relevant agents using a two-stage
        "gating and ranking" process, now including an `is_parked` flag.
        """
        num_agents_to_keep = self.features_config['num_agents']
        # Feature: [x, y, vx, vy, cos(h), sin(h), length, width, is_vehicle, is_ped_or_cyc, is_parked]
        feature_dim = 11
        
        agent_features = np.zeros((num_agents_to_keep, feature_dim), dtype=np.float32)
        agent_mask = np.zeros(num_agents_to_keep, dtype=bool)

        if other_agents_ego.shape[0] == 0:
            return agent_features, agent_mask

        # --- Stage 1: Gating (Filter by Relevance Box) ---
        max_lon = self.relevance_config['max_longitudinal_dist']
        max_lon_behind = self.relevance_config['max_longitudinal_dist_behind']
        max_lat = self.relevance_config['max_lateral_dist']

        # Create boolean masks based on ego-centric coordinates
        lon_mask = (other_agents_ego[:, 0] > -max_lon_behind) & (other_agents_ego[:, 0] < max_lon)
        lat_mask = np.abs(other_agents_ego[:, 1]) < max_lat
        
        relevant_mask = lon_mask & lat_mask
        
        relevant_agents = other_agents_ego[relevant_mask]
        relevant_types = other_agent_types[relevant_mask]
        relevant_original_indices = original_indices[relevant_mask] # Keep track of indices

        if relevant_agents.shape[0] == 0:
            return agent_features, agent_mask

        # --- Stage 2: Ranking (Sort relevant agents by proximity) ---
        distances = np.linalg.norm(relevant_agents[:, :2], axis=1)
        
        # Determine how many agents to finally select
        num_to_take = min(len(distances), num_agents_to_keep)
        
        # Get the indices of the closest agents *within the relevant set*
        nearest_indices_in_relevant_set = np.argsort(distances)[:num_to_take]
        
        # Select the final set of agents and their types
        final_agents = relevant_agents[nearest_indices_in_relevant_set]
        final_types = relevant_types[nearest_indices_in_relevant_set]
        final_original_indices = relevant_original_indices[nearest_indices_in_relevant_set]

        # --- Populate Feature Matrix (same as before, but with `final_agents`) ---
        agent_features[:num_to_take, 0:2] = final_agents[:, [0, 1]]  # x, y
        agent_features[:num_to_take, 2:4] = final_agents[:, [7, 8]]  # vx, vy
        headings = final_agents[:, 6]
        agent_features[:num_to_take, 4] = np.cos(headings)
        agent_features[:num_to_take, 5] = np.sin(headings)
        agent_features[:num_to_take, 6:8] = final_agents[:, [3, 4]] # length, width

        type_one_hot = np.zeros((num_to_take, 2), dtype=np.float32)
        type_one_hot[final_types == 1, 0] = 1.0                # is_vehicle
        type_one_hot[np.isin(final_types, [2, 3]), 1] = 1.0    # is_ped_or_cyc
        agent_features[:num_to_take, 8:10] = type_one_hot
        
        is_parked_status = is_parked_mask_full[final_original_indices]
        agent_features[:num_to_take, 10] = is_parked_status.astype(np.float32)
        
        agent_mask[:num_to_take] = True

        agent_mask[:num_to_take] = True
        return agent_features, agent_mask

    def _get_map_features(self, lane_polylines_ego: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        """
        --- UPGRADED METHOD ---
        Extracts features for the M most relevant lane centerlines using a two-stage
        "gating and ranking" process.
        """
        num_lanes_to_keep = self.features_config['num_map_polylines']
        points_per_lane = self.features_config['map_points_per_polyline']
        
        map_features = np.zeros((num_lanes_to_keep, points_per_lane, 2), dtype=np.float32)
        map_mask = np.zeros(num_lanes_to_keep, dtype=bool)

        if not lane_polylines_ego:
            return map_features, map_mask

        non_empty_polylines = [p for p in lane_polylines_ego if p.shape[0] > 0]
        if not non_empty_polylines:
            return map_features, map_mask

        # --- Stage 1: Gating (Filter by Relevance Box) ---
        # Reuse the same relevance parameters defined for agents
        max_lon = self.relevance_config['max_longitudinal_dist']
        max_lon_behind = self.relevance_config['max_longitudinal_dist_behind']
        max_lat = self.relevance_config['max_lateral_dist']
        
        gated_polylines = []
        for p in non_empty_polylines:
            # Get the bounding box of the polyline
            min_x, max_x = np.min(p[:, 0]), np.max(p[:, 0])
            min_y, max_y = np.min(p[:, 1]), np.max(p[:, 1])

            # Check for overlap between the polyline's bounding box and our relevance box
            # If the polyline is completely outside the box, discard it
            if (max_x < -max_lon_behind or min_x > max_lon or
                max_y < -max_lat or min_y > max_lat):
                continue
            gated_polylines.append(p)
        
        if not gated_polylines:
            return map_features, map_mask

        # --- Stage 2: Ranking (Sort by true minimum distance) ---
        # For each gated polyline, find the distance to its closest point from the ego vehicle (origin)
        distances = [np.linalg.norm(p[:, :2], axis=1).min() for p in gated_polylines]

        # Determine how many polylines to finally select
        num_to_take = min(len(distances), num_lanes_to_keep)

        # Get the indices of the closest polylines *within the gated set*
        nearest_indices = np.argsort(distances)[:num_to_take]

        # --- Populate Feature Matrix ---
        for i, original_idx in enumerate(nearest_indices):
            polyline = gated_polylines[original_idx]
            resampled = geometry.resample_polyline(polyline[:, :2], points_per_lane)
            map_features[i, :, :] = resampled
        
        map_mask[:num_to_take] = True
        return map_features, map_mask

    def _get_traffic_light_features(self, traffic_lights_global: np.ndarray, ego_pose: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Extracts a feature vector for the single most relevant traffic light."""
        # Feature: [is_red_ahead, dist_to_stop_line]
        tl_features = np.zeros(2, dtype=np.float32)
        # The feature is always considered "valid" even if it's all zeros.
        tl_mask = np.array([True], dtype=bool)
        
        if traffic_lights_global.shape[0] == 0:
            return tl_features, tl_mask

        stop_points_global = traffic_lights_global[:, 2:4]
        distances = np.linalg.norm(stop_points_global - ego_pose[:2], axis=1)
        closest_light_idx = np.argmin(distances)
        closest_light_state = traffic_lights_global[closest_light_idx]
        
        # Check if the light is in a "STOP" state (red arrow, red light, or flashing red)
        is_red = int(closest_light_state[1]) in {1, 4, 7}
        
        # To be relevant, the stop point must be generally in front of the SDC (positive x in ego frame)
        stop_point_ego = geometry.transform_points(closest_light_state[2:4].reshape(1, 2), ego_pose)
        
        if is_red and stop_point_ego[0, 0] > -1.0: # Use -1m buffer for robustness
            tl_features[0] = 1.0 # is_red_light_ahead flag
            tl_features[1] = stop_point_ego[0, 0] # Use longitudinal distance
            
        return tl_features, tl_mask
        
    def _get_goal_features(self, sdc_route: np.ndarray, current_timestep: int, ego_pose: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Extracts future waypoints from the expert's route as the goal."""
        num_goal_points = self.features_config.get('num_goal_points', 5)
        horizon_seconds = np.arange(1, num_goal_points+1)
        horizon_steps = (horizon_seconds * 10).astype(int)
        # Feature: just the ego-centric (x, y) coordinates of the goal points
        feature_dim = 2
        
        goal_features = np.zeros((num_goal_points, feature_dim), dtype=np.float32)
        goal_mask = np.zeros(num_goal_points, dtype=bool)

        target_timesteps = current_timestep + horizon_steps
        # Create a boolean mask of which future timesteps are valid (i.e., within the scenario bounds)
        valid_mask = target_timesteps < sdc_route.shape[0]
        
        valid_target_timesteps = target_timesteps[valid_mask]
        if len(valid_target_timesteps) == 0:
            return goal_features, goal_mask

        # Get global waypoints and transform them to the ego-centric frame
        future_waypoints_global = sdc_route[valid_target_timesteps, :2]
        future_waypoints_ego = geometry.transform_points(future_waypoints_global, ego_pose)
        
        # Populate the feature tensor only for the valid future points
        goal_features[valid_mask, :] = future_waypoints_ego
        goal_mask[valid_mask] = True
        
        return goal_features, goal_mask
    
    def _classify_parked_vehicles(self, source_data: Dict) -> np.ndarray:
        """
        Creates a boolean mask to identify parked vehicles across the entire scenario.
        
        Returns:
            A boolean mask of shape (total_num_agents,) where True means the agent IS parked.
        """
        all_trajs = source_data['all_agent_trajectories'] # (num_agents, num_timesteps, 9)
        valid_masks = source_data['valid_mask']           # (num_agents, num_timesteps)
        
        speeds = np.linalg.norm(all_trajs[:, :, 7:9], axis=-1)
        speed_thresh = self.features_config.get('parked_vehicle_speed_threshold', 0.5)
        
        # An agent is considered parked if its speed is ALWAYS below the threshold
        # for every single timestep where it is valid.
        is_slow = speeds < speed_thresh
        
        # By using (is_slow | ~valid_masks), we ensure that invalid timesteps don't
        # cause the `np.all` to fail. We only care about the valid timesteps.
        always_slow_or_invalid = np.all(is_slow | ~valid_masks, axis=1)
        
        # A parked agent must have at least one valid timestep to be considered.
        has_any_valid_step = np.any(valid_masks, axis=1)
        
        is_parked_mask = always_slow_or_invalid & has_any_valid_step
        return is_parked_mask