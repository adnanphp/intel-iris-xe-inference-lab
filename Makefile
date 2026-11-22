# Reproducible workflow targets for the Intel Iris Xe Inference Lab.
# Usage: make <target>

PYTHON ?= python3
FD_MODEL := model/face-detection-retail-0004.xml
MN_MODEL := model/mobilenet-v3-large-1.0-224-tf.xml
MN_INT8  := model/mobilenet-v3-large-int8.xml

.PHONY: help setup sweep parallel plots compare quantize app test clean

help:
	@echo "Targets:"
	@echo "  setup     - install Python dependencies"
	@echo "  sweep     - full CPU/GPU batch-size sweep (face-detection)"
	@echo "  parallel  - sync/async/multiprocess comparison (face-detection)"
	@echo "  quantize  - INT8-quantize the MobileNet model"
	@echo "  plots     - regenerate all plots"
	@echo "  compare   - cross-model + cross-precision comparison table + plot"
	@echo "  app       - run batch inference on test_images/"
	@echo "  test      - run the pytest suite"
	@echo "  clean     - remove generated results/plots (keeps CSVs)"

setup:
	$(PYTHON) -m pip install -r requirements.txt

sweep:
	@echo "==> face-detection CPU/GPU sweep"
	$(PYTHON) benchmarks/sweep.py --csv results/sweep_python.csv

parallel:
	@echo "==> sync / async / multiprocess"
	rm -f results/parallel_results.csv
	$(PYTHON) benchmarks/parallel_benchmark.py --model $(FD_MODEL) --device GPU --mode sync --batch-size 1 --iterations 100
	for d in 2 4 8; do \
	  $(PYTHON) benchmarks/parallel_benchmark.py --model $(FD_MODEL) --device GPU --mode async --batch-size 1 --iterations 100 --workers $$d; \
	done
	for w in 1 2 4 6 8; do \
	  $(PYTHON) benchmarks/parallel_benchmark.py --model $(FD_MODEL) --device GPU --mode multiprocess --batch-size 1 --iterations 50 --workers $$w; \
	done

quantize:
	@echo "==> INT8 quantization of MobileNet"
	$(PYTHON) benchmarks/quantize_model.py --model $(MN_MODEL) --output $(MN_INT8) --input-shape 1,3,224,224 --calib-samples 100

plots:
	@echo "==> regenerating plots"
	$(PYTHON) benchmarks/plot_results.py results/sweep_python.csv
	$(PYTHON) benchmarks/plot_parallel.py results/parallel_results.csv

compare:
	@echo "==> cross-model comparison"
	$(PYTHON) benchmarks/compare_models.py \
	  --csv results/sweep_python.csv \
	        results/mobilenet_results.csv \
	        results/mobilenet_int8_results.csv

app:
	@echo "==> batch inference on test_images/"
	$(PYTHON) app/batch_inference.py \
	  --input test_images \
	  --model $(FD_MODEL) \
	  --device GPU --batch-size 4 \
	  --output output

test:
	$(PYTHON) -m pytest tests/ -v

clean:
	rm -rf results/plots/*.png
	@echo "Removed generated plots (CSVs kept)."
