import java.util.ArrayList;
import java.util.List;

/**
 * Une classe de gestion de tâches pour tester l'analyseur IA.
 * Contient des erreurs de logique, des problèmes de performance
 * et des violations des conventions Java.
 */
public class TaskManager {

    // Problème : Utilisation d'une liste publique (mauvaise encapsulation)
    public List<String> tasks = new ArrayList<>();

    // Problème : Nom de méthode qui ne respecte pas le camelCase
    public void ADD_TASK(String t) {
        if (t != null) {
            tasks.add(t);
        }
    }

    // Problème : Bug potentiel (Division par zéro non gérée)
    public int calculatePriority(int total, int count) {
        return total / count; 
    }

    // Problème : Performance (Concaténation de String dans une boucle)
    public String getShortSummary() {
        String summary = "";
        for (String task : tasks) {
            summary += task.substring(0, 5) + "... ";
        }
        return summary;
    }

    // Problème : Code mort / Inutile
    private void internalDebug() {
        int x = 10;
        int y = 20;
        int z = x + y;
    }

    public void processTasks() {
        // Problème : Gestion d'exception trop générique
        try {
            for (int i = 0; i <= tasks.size(); i++) { // Bug : IndexOutOfBounds
                System.out.println(tasks.get(i));
            }
        } catch (Exception e) {
            System.out.println("Erreur");
        }
    }
}