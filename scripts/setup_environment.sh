#!/usr/bin/env bash
set -euo pipefail

PYTHON_VERSION="3.13.5"
VENV_DIR=".venv"

cd "$(dirname "$0")/.."

command_exists() {
	command -v "$1" >/dev/null 2>&1
}

find_python() {
	if command_exists "python${PYTHON_VERSION}"; then
		command -v "python${PYTHON_VERSION}"
		return 0
	fi

	if command_exists python3; then
		local version
		version="$(python3 -c 'import platform; print(platform.python_version())')"
		if [ "$version" = "$PYTHON_VERSION" ]; then
			command -v python3
			return 0
		fi
	fi

	if command_exists python; then
		local version
		version="$(python -c 'import platform; print(platform.python_version())' 2>/dev/null || true)"
		if [ "$version" = "$PYTHON_VERSION" ]; then
			command -v python
			return 0
		fi
	fi

	return 1
}

install_python_with_pyenv() {
	if ! command_exists pyenv; then
		echo "[ERROR] Python ${PYTHON_VERSION} bulunamadi ve pyenv kurulu degil."
		echo "Once pyenv kur:"
		echo "  brew install pyenv"
		echo "Sonra tekrar calistir:"
		echo "  bash scripts/setup_environment.sh"
		exit 1
	fi

	if ! pyenv versions --bare | grep -qx "$PYTHON_VERSION"; then
		echo "[INFO] Python ${PYTHON_VERSION} pyenv ile kuruluyor..."
		pyenv install "$PYTHON_VERSION"
	fi

	pyenv local "$PYTHON_VERSION"
	pyenv which python
}

PYTHON_BIN="$(find_python || install_python_with_pyenv)"

echo "[INFO] Kullanilan Python: $("$PYTHON_BIN" --version)"

if [ ! -d "$VENV_DIR" ]; then
	echo "[INFO] Sanal ortam olusturuluyor: ${VENV_DIR}"
	"$PYTHON_BIN" -m venv "$VENV_DIR"
fi

VENV_PYTHON="${VENV_DIR}/bin/python"

echo "[INFO] pip guncelleniyor..."
"$VENV_PYTHON" -m pip install --upgrade pip

echo "[INFO] requirements.txt kuruluyor..."
"$VENV_PYTHON" -m pip install -r requirements.txt

echo
echo "[OK] Kurulum tamamlandi."
echo "Aktif etmek icin:"
echo "  source ${VENV_DIR}/bin/activate"
echo
echo "Ornek calistirma:"
echo "  python scripts/run_autoencoder.py --dataset-name breast_cancer_data.csv"
