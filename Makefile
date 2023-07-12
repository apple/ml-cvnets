SHELL := /bin/bash
YEAR = $$(date +%Y)
COLOR_LOG = \033[34m
COLOR_INFO = \033[32m
COLOR_WARNING = \033[33m
COLOR_ERROR = \033[31m
COLOR_END = \033[0m

.PHONY: format install-githooks prepush-check

format:
	# Checking missing license headers...
	@for f in $$(find . -iname "*.py"); do \
	    [ -s "$$f" ] || continue; \
	    (head -3 $$f | grep -q "# *Copyright (C) .* Apple Inc. All Rights Reserved.") && continue; \
	    [[ "$$CHECK_ONLY" == "1" ]] && exit 1; \
	    sed -i '' -e "1s/^/#\n# For licensing see accompanying LICENSE file.\n# Copyright (C) $(YEAR) Apple Inc. All Rights Reserved.\n#\n\n/" $$f; \
	    printf "$(COLOR_LOG)Added license header for $$f$(COLOR_END)\n"; \
	done

	# Running isort...
	@if [[ "$$CHECK_ONLY" == "1" ]]; then \
	    isort --check-only .; \
	else \
	    isort .; \
	fi

	# Running black formatter...
	@if [[ "$$CHECK_ONLY" == "1" ]]; then \
	    black --check .; \
	else \
	    black .; \
	fi

	@if [[ "$$CHECK_ONLY" == "1" ]]; then \
	    echo "# Checking coding conventions..."; \
	    convention_test_files=(tests/test_conventions.py); \
	    [[ -d tests/internal ]] && convention_test_files+=(tests/internal/test_internal_conventions.py); \
	    if ! pytest --junit-xml="" -q "$${convention_test_files[@]}"; then \
	        printf "$(COLOR_ERROR) Please manually fix the above convention errors. $(COLOR_END)\n"; \
	        exit 1; \
	    fi; \
	fi

	# Formatting checks succeeded.

prepush-check:
	@printf "$(COLOR_LOG)[pre-push hook]$(COLOR_END)\n"
	@CHECK_ONLY=1 make format || (printf "$(COLOR_ERROR)Formatting checks failed.$(COLOR_END) Please run '$(COLOR_INFO)make format$(COLOR_END)' command, commit, and push again.\n" && exit 1);
	@if [ -n "$$(git status --porcelain)" ]; then \
	    printf "$(COLOR_WARNING)Formatting checks succeeded, but please consider committing UNCOMMITTED changes to the following files:$(COLOR_END)\n"; \
	    git status --short; \
	else \
	    printf "$(COLOR_INFO)Formatting checks succeeded.$(COLOR_END)\n"; \
	fi

install-githooks:
	@echo -e "#!/bin/bash\n" '\
	    printf "$(COLOR_LOG)[pre-push hook]$(COLOR_END) Running formatting checks and fixes... To skip this hook, please run \"git push --no-verify\".\n"; \
	    set -e; \
	    if grep -q "^prepush-check:" Makefile 2>/dev/null; then \
	        make prepush-check; \
	    else \
	        printf "$(COLOR_WARNING)WARNING:$(COLOR_END) Skipping the pre-push formatting checks and fixes. The git hook is installed (probably on a different git branch), but Makefile is either missing or old on this branch.\n"; \
	    fi \
	' > "$$(git rev-parse --git-path hooks)/pre-push"
	chmod +x "$$(git rev-parse --git-path hooks)/pre-push"
	# Successfully installed the pre-push hook.

test-all:
# Run all tests and set OMP/MKL threads to one to allow pytest
# parallelization
	MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 OMP_NUM_THREADS=1 pytest . -n auto $(extra_args)

coverage-test-all:
	MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 OMP_NUM_THREADS=1 coverage run -m pytest . -n auto $(extra_args)
