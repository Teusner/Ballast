SRC_DIR = src/*.py
IMGS_DIR = img
VIDEO_DIR = video
PLOTS_DIR = data

FPS = 25

# Directory guard
dir_guard = @mkdir -p $(@D)

all : plots images videos

# Images
images: $(IMGS_DIR)/*.png

$(IMGS_DIR)/*.png: $(SRC_DIR)
	$(dir_guard)
	python3 src/main.py --imgs-path $(IMGS_DIR) --fps $(FPS)

# Plots
plots: $(PLOTS_DIR)/*.png

$(PLOTS_DIR)/*.png: $(SRC_DIR)
	$(dir_guard)
	python3 src/main.py --plots-path $(PLOTS_DIR) --plot-angles --plot-depth --plot-ballasts --plot-vertical-velocity --plot-metacenter --plot-wrench --fps $(FPS)

# Videos
videos: $(VIDEO_DIR)/out.mp4

$(VIDEO_DIR)/out.mp4: $(IMGS_DIR)/*.png
	$(dir_guard)
	ffmpeg -framerate $(FPS) -pattern_type glob -i '$(IMGS_DIR)/*.png' -c:v libx264 -pix_fmt yuv420p $(VIDEO_DIR)/out.mp4 -y

clean:
	rm -rf src/__pycache__
	rm -rf $(IMGS_DIR)
	rm -rf $(VIDEO_DIR)
	rm -rf $(PLOTS_DIR)
