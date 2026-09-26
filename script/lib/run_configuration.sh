#!/usr/bin/env bash

# Helpers for replaying output-affecting settings from a persistent run-config.
# Explicit caller environment always wins so controlled A/B tests can override
# exactly one setting while inheriting every other recorded setting.

jasna_recorded_configuration_value() {
  local configuration_path="$1"
  local key="$2"
  [[ "$key" =~ ^[a-z0-9_]+$ && -s "$configuration_path" ]] || return 1
  /usr/bin/awk -F= -v key="$key" '$1 == key { print substr($0, index($0, "=") + 1); exit }' \
    "$configuration_path"
}

jasna_adopt_recorded_environment() {
  local configuration_path="$1"
  local key="$2"
  local environment_name="$3"
  [[ "$environment_name" =~ ^JASNA_[A-Z0-9_]+$ ]] || return 2
  /usr/bin/printenv "$environment_name" >/dev/null 2>&1 && return 0
  local recorded_value
  recorded_value="$(
    jasna_recorded_configuration_value "$configuration_path" "$key" || true
  )"
  [[ -n "$recorded_value" ]] || return 0
  export "$environment_name=$recorded_value"
}
