# facial_recognition_simpsons

Proiect in cadrul cursului CAVA - UB, FMI, an universitar 2021-2022, student Stanciu Andrei Calin </br>
</br>
Instructiuni rulare:</br>
    * codul sursa pentru realizarea predictiilor se afla in evaluare/make_predictions.py</br>
    * codul sursa pentru evaluarea predictiilor facute de script-ul anterior se afla in
    evaluare/evalueaza_solutie.py
    * se ruleaza pe rand, fara nici un argument: "python3 make_predictions.py" , "python3 evalueaza_solutie.py"</br>
        direct din folderul evaluare (pentru a functiona path-urile relative)</br>
    * path-urile sunt configurate pentru linux, este posibil ca pentru windows/ orice alt sistem de operare, sa trebuiasca sa fie schimbate </br>
        (din make_predictions.py, variabilele globale TEST_DATA_PATH, PREDICTION_PATH_TASK1, PREDICTION_PATH_TASK2</br>
                                                        si eventual TRAIN_DATA_PATH)</br>
        (din evalueaza_solutie.py, variabilele globale solution_path_root, ground_truth_path_root)</br>
    * python 3.8.10, opencv-python (4.5.3.56), numpy (1.20.3), nici o alta dependinta</br>
    * codul face predictii pentru datele aflate in folderul testare/simpsons_testare</br>
    * codul genereaza predictiile in formatul cerut in fisiere_solutie/Stanciu_Calin_331/task1 </br>
       respectiv in fisiere_solutie/Stanciu_Calin_331/task2 </br>
</br>
Alte informatii despre fisiere:</br>
    * codul din make_predictions.py, by default, incarca modelele pentru cele doua task uri deja antrenate, </br>
         din model_task1 si model_task2
    * am inclus si datele de antrenare, iar codul din make_predictions.py poate fi modificat </br>
        pentru a antrena pe loc cele doua resnet, (in make_predictions.py: task1_run(load=False) si/sau task2_run(load=False))
    * in evaluare/templates/ se afla template urile (extrase semi-manual din datele de antrenare) pentru cifre/contur   </br>
       folosite de catre evalueaza_solutie.py</br>
    * in evaluare/aux_py/ se afla bucati de cod folosite in timpul rezolvarii temei, </br>
       de exemplu abordarea folosind svm cu hog, sau pentru gridsearch </br>
       - aceste bucati de cod au fost introduse in scop demonstrativ, fara a fi folosite </br>
        	in mod direct pentru generarea predictiilor de catre make_predictions.py sau evalueaza_solutie.py</br>
