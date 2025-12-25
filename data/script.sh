for scene in ./x_rgb/*; do
  scene_name=$(basename "$scene")
  rgb_dir="$scene/image_02/data"

  if [ -d "$rgb_dir" ]; then
    for img in "$rgb_dir"/*.png; do
      base=$(basename "$img")
      cp "$img" "./images/${scene_name}_${base}"
    done
  fi
done

