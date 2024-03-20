To run the script,\
`python georeference.py --images <path_to_image_folder> --poses <pose_csv_file>`

If the images are decompressed to `./images` and the pose file can be found as `./lab3_pose.csv` simply run,\
`python georeference.py`

The script solves the landmark estimation problem in the following steps,
1. Read all poses
2. For each image,
 - find the landmark observation in image using Hough circle finder.
 - Reproject 2D observation to 3D position (taking advantage of the fact that all landmark lies on the xy plane in inertial frame)
 - track landmark observation, position, and associated poses.
3. Cluster all position into groups of landmark.
 - Reject cluster with inferior number of observations.
4. Formulate least square problem for each landmark aiming to reduce reprojection error.
5. Plot result and show final position estimate for landmarks
