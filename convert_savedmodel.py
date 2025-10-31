#!/usr/bin/env python3
"""
Script para convertir SavedModel a TensorFlow.js
"""

import os
import sys
import tensorflowjs as tfjs

def convert_savedmodel_to_tfjs():
    """
    Convierte el SavedModel a formato TensorFlow.js
    """
    
    input_path = 'frutas_cnn_savedmodel'
    output_path = 'frutas_cnn_savedmodel_tfjs'
    
    if not os.path.exists(input_path):
        print(f"❌ Error: No se encuentra la carpeta {input_path}")
        return False
    
    try:
        print("🔄 Convirtiendo SavedModel a TensorFlow.js...")
        print(f"📂 Entrada: {input_path}")
        print(f"📁 Salida: {output_path}")
        
        # Crear directorio de salida
        os.makedirs(output_path, exist_ok=True)
        
        # Convertir SavedModel a TensorFlow.js
        tfjs.converters.convert_tf_saved_model(
            input_path,
            output_path
        )
        
        print("✅ Conversión exitosa!")
        
        # Verificar archivos generados
        model_json_path = os.path.join(output_path, 'model.json')
        if os.path.exists(model_json_path):
            print(f"✅ model.json generado en: {model_json_path}")
            
            # Contar archivos .bin
            bin_files = [f for f in os.listdir(output_path) if f.endswith('.bin')]
            print(f"✅ {len(bin_files)} archivos de pesos generados")
            
            return True
        else:
            print("❌ No se generó model.json")
            return False
            
    except Exception as e:
        print(f"❌ Error durante la conversión: {str(e)}")
        return False

if __name__ == "__main__":
    print("🍎 Conversor SavedModel a TensorFlow.js")
    print("=" * 40)
    
    # Verificar dependencias
    try:
        import tensorflowjs
        print("✅ TensorFlow.js encontrado")
    except ImportError:
        print("❌ TensorFlow.js no está instalado")
        print("Instálalo con: pip install tensorflowjs")
        sys.exit(1)
    
    success = convert_savedmodel_to_tfjs()
    
    if success:
        print("\n🎉 ¡Conversión completa!")
        print("\n📋 Instrucciones:")
        print("1. El modelo convertido está en 'frutas_cnn_savedmodel_tfjs'")
        print("2. Actualiza MODEL_PATH en script.js si es necesario")
    else:
        print("\n💥 La conversión falló")
        sys.exit(1)