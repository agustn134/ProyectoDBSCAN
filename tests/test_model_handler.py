import sys
from pathlib import Path

# Agregar la raíz del proyecto al path para poder importar utils
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.model_handler import min_samples_recomendado

def test_min_samples_muestra_chica():
    # Para muestras pequeñas (~73) debería usar el piso clásico (D+1 = 6)
    assert min_samples_recomendado(73) == 6

def test_min_samples_muestra_mediana():
    # Para muestras donde el 1% es <= 6, se debe mantener en 6
    assert min_samples_recomendado(500) == 6

def test_min_samples_muestra_grande():
    # Para muestras muy grandes (5000), el 1% es 50, que es el techo
    assert min_samples_recomendado(5000) == 50

def test_min_samples_nunca_baja_del_piso():
    # Incluso con muy pocos registros, debe ser mínimo 6
    assert min_samples_recomendado(10) >= 6

def test_min_samples_nunca_pasa_el_techo():
    # Incluso con datasets gigantes, no debe pasar de 50
    assert min_samples_recomendado(100000) == 50
