import time
import subprocess
import sys
import os

def run_simulation_cycle(simulador_path, ia_path):
    """
    Executa um ciclo completo de simulação.
    Roda o simulador, depois a IA.
    """
    print("Iniciando novo ciclo de simulação...")
    
    # --- Executa o simulador Gemini (bb.py) ---
    print(f"[{time.strftime('%H:%M:%S')}] Rodando o simulador ({os.path.basename(simulador_path)})...")
    try:
        # Usa subprocess para rodar o script Python
        result = subprocess.run([sys.executable, simulador_path], check=True, capture_output=True, text=True)
        print("Simulador concluído. Saída:")
        print(result.stdout)
        if result.stderr:
            print("Erros do simulador:")
            print(result.stderr)
    except subprocess.CalledProcessError as e:
        print(f"Erro ao rodar o simulador: {e}")
        print(f"Saída do erro: {e.stderr}")
        return False # Aborta o ciclo em caso de falha
    
    time.sleep(1) # Pequena pausa entre os scripts

    # --- Executa a IA Adla (adla.py) ---
    print(f"[{time.strftime('%H:%M:%S')}] Rodando a IA ({os.path.basename(ia_path)})...")
    try:
        result = subprocess.run([sys.executable, ia_path], check=True, capture_output=True, text=True)
        print("IA concluída. Saída:")
        print(result.stdout)
        if result.stderr:
            print("Erros da IA:")
            print(result.stderr)
    except subprocess.CalledProcessError as e:
        print(f"Erro ao rodar a IA: {e}")
        print(f"Saída do erro: {e.stderr}")
        return False # Aborta o ciclo em caso de falha

    print("Ciclo de simulação concluído com sucesso.")
    return True

def main_loop():
    """
    Função principal que executa os ciclos em loop.
    """
    # Substitua 'bb.py' e 'adla.py' pelos nomes dos seus arquivos se forem diferentes
    simulador_file = 'bb.py'
    ia_file = 'adla.py'
    
    if not os.path.exists(simulador_file) or not os.path.exists(ia_file):
        print(f"Erro: Um ou mais arquivos ({simulador_file} ou {ia_file}) não foram encontrados no diretório atual.")
        return

    while True:
        cycle_start_time = time.time()
        
        success = run_simulation_cycle(simulador_file, ia_file)
        
        cycle_duration = time.time() - cycle_start_time
        wait_time = 5 - cycle_duration
        
        if wait_time > 0:
            print(f"Aguardando {wait_time:.2f} segundos antes do próximo ciclo...")
            time.sleep(wait_time)
        else:
            print("O ciclo de simulação demorou mais que 5 segundos. Iniciando o próximo imediatamente.")
            
        print("-" * 70)

if __name__ == "__main__":
    main_loop()

