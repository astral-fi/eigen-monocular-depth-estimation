for scene in ./y_depth/*; do
  scene_name=$(basename "$scene")
  depth_dir="$scene/proj_depth/groundtruth/image_02"

  if [ -d "$depth_dir" ]; then
    for d in "$depth_dir"/*.png; do
      base=$(basename "$d")
      cp "$d" "./depths/${scene_name}_${base}"
    done
  fi
done

