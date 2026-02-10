import os
import sys
from src.agents.code_analyzer import CodeAnalyzerAgent


def main():

    print(" LLM CODE ANALYZER ")

    agent = CodeAnalyzerAgent()

    print("Que voulez-vous analyser ?")
    print("1  Fichier")
    print("2  Projet")
    print("3  Coller du code")

    choice = input("\nVotre choix (1/2/3): ").strip()

   
    if choice == "1":
        path = input("\nEntrez le chemin du fichier:\n> ").strip().strip('"')

        if not os.path.isfile(path):
            print(" Fichier invalide.")
            return

        result = agent.analyze_file(path)
        agent.print_analysis_summary(result)

   
    elif choice == "2":
        path = input("\n Entrez le chemin du dossier:\n> ").strip().strip('"')

        if not os.path.isdir(path):
            print(" Dossier invalide.")
            return

        results = agent.analyze_directory(path)

        for r in results:
            agent.print_analysis_summary(r)

        print(agent.generate_summary_report(results))

   
    elif choice == "3":
        lang = input("\nLangage (python/javascript/java): ").strip()

        print("\n Collez votre code maintenant.")
        print("Terminez avec CTRL+Z puis ENTER (Windows)\n")

        code = sys.stdin.read()

        result = agent.analyze_code(code, lang)
        agent.print_analysis_summary(result)

    else:
        print(" Choix invalide.")


if __name__ == "__main__":
    main()
