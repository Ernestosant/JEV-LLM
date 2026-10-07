# Resultados JEV-LLM v3

500 problemas de test; 6000 inferencias y 1800 mediciones de latencia.
Las semillas no son problemas independientes. El baseline principal es el modelo de 13B greedy.

## Comparacion Principal

- Diferencia de exactitud JFINAL - B13_GREEDY: 10.7333 puntos porcentuales.
- IC 97.5% de exactitud: [7.533333333333333, 14.133333333333333].
- Speedup geometrico pareado: 1.00192; mayor que 1 favorece al hibrido.
- IC 97.5% de speedup: [0.883867813624088, 1.1264255001970094].

Los IC coprincipales usan bootstrap por problema, estratificado por dominio, y ajuste Bonferroni.
La latencia usa 100 problemas y la mediana de nueve mediciones por sistema y problema.

## Exactitud

| Condicion | Inferencias | Problemas | Exactitud % |
|---|---|---|---|
| G_SINGLE | 1500 | 500 | 91 |
| JFINAL | 1500 | 500 | 94.7333 |
| B13 | 1500 | 500 | 84.2 |
| G_GREEDY | 500 | 500 | 92.2 |
| B13_GREEDY | 500 | 500 | 84 |
| Q9_GREEDY | 500 | 500 | 89.2 |
| FIXED_BRANCH0_SHARED | 1500 | 500 | 90.7333 |
| VOTE4SHARED | 1500 | 500 | 94.6 |

## Conclusiones Verificadas

```json
{
  "quality_superiority": true,
  "latency_superiority": false,
  "joint_superiority": false,
  "quality_practical_point_5pp": true,
  "latency_practical_point_10pct_reduction": false,
  "rule": "Quality lower97.5 > 0; latency lower97.5 > 1; joint requires both. Practical thresholds are point estimates."
}
```

El balance es por nivel de fuente, no por dificultad intrinseca. La revision fue agentica, no humana.
Ver report.md y analysis.json para controles, contrastes secundarios, auditoria y limitaciones.
