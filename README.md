# SIwRProject
## Użyta metoda
Do śledzenia bazuje na Bipartite graph który jest optymalizowany przy pomocy funkcji linear_sum_assignment() z biblioteki scipy.
Macierz opisująca graf jest budowana poprzez średnią ważoną następujących wskaźników podobieństwa obrazów:
1. Structural Similarity Index Measure (SSIM)
2. Template Matching
3. Stosunek pola bounding boxów
4. Porównania Histogramów
5. Intersection Over Union
