# checkmake: ignore-rule maxbodylength
# ──────────────────────────────────────────────────────────────────────────────
#  Configuration & Variables
# ──────────────────────────────────────────────────────────────────────────────

# Colors & symbols
NO_COLOR    := \033[0m
ERROR       := \033[0;31m
SUCCESS     := \033[0;32m
WARNING     := \033[0;33m
INFO        := \033[0;36m
UNDERLINE   := \033[4m
BOLD        := \033[1m

OK      := $(SUCCESS)✓$(NO_COLOR)
FAIL    := $(ERROR)✗$(NO_COLOR)
ARROW   := $(INFO)→$(NO_COLOR)

# Variables
VENV_DIR := .venv

.DEFAULT_GOAL := help

# ─────────────────────────────────────────────────────────────────────────────
# Help / Default target
# ─────────────────────────────────────────────────────────────────────────────
# checkmake: ignore-rule maxbodylength
.PHONY: help
help: ## Show this help message
	@echo "$(BOLD)Available Makefile commands$(NO_COLOR)"
	@echo "Usage: make $(UNDERLINE)target$(NO_COLOR)"
	@echo ""
	@grep -H -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | sort | \
	awk 'BEGIN { \
		FS = ":.*?## "; \
		printf "  $(BOLD)%-25s %-20s %-35s$(NO_COLOR)\n", "COMMAND", "FILE", "EXPLANATION"; \
		printf "  %-25s %-20s %-35s\n", "-------", "----", "-----------" \
	} \
	{ \
		split($$0, a, ":"); \
		file = a[1]; \
		sub("^make/", "", file); \
		sub("\\.mk$$", "", file); \
		target = a[2]; \
		help = $$2; \
		printf "  $(INFO)%-25s$(NO_COLOR) %-20s %-35s\n", target, file, help \
	}'
