# JEV-LLM v3 Offline Analysis

General ~13B greedy is the fixed primary baseline.

500 test problems; 6000 physical cases. Seeds are not additional independent problems.

## Source Sampling And Intrinsic Difficulty

Balance is by original source sampling tier within the reviewed mathematical domain, not by intrinsic difficulty.
Test per domain:30/40/30 source tiers; pilot:3/4/3; latency:6/8/6. Source tier hard does not mean genuinely hard.
550 selected test/pilot problems are agent-reviewed; human_reviewed=false. Unused source candidates do not enter these counts.
The bootstrap remains DOMAIN-stratified; no source-tier or intrinsic-difficulty tuning of co-primary effects.

| Split | Problems | Source-Tier Easy | Source-Tier Medium | Source-Tier Hard | Intrinsic Easy | Intrinsic Medium | Intrinsic Hard |
|---|---|---|---|---|---|---|---|
| test | 500 | 150 | 200 | 150 | 394 | 96 | 10 |
| pilot | 50 | 15 | 20 | 15 | 38 | 11 | 1 |
| latency | 100 | 30 | 40 | 30 | 78 | 21 | 1 |

Per-domain source-tier and intrinsic counts:
```json
{
  "quota_axis": "source_sampling_tier",
  "splits": {
    "test": {
      "quota_axis": "source_sampling_tier",
      "n_problems": 500,
      "source_sampling_tier_counts": {
        "easy": 150,
        "medium": 200,
        "hard": 150
      },
      "intrinsic_difficulty_counts": {
        "easy": 394,
        "medium": 96,
        "hard": 10
      },
      "by_domain": {
        "algebra": {
          "source_sampling_tier_counts": {
            "easy": 30,
            "medium": 40,
            "hard": 30
          },
          "intrinsic_difficulty_counts": {
            "easy": 76,
            "medium": 22,
            "hard": 2
          }
        },
        "arithmetic": {
          "source_sampling_tier_counts": {
            "easy": 30,
            "medium": 40,
            "hard": 30
          },
          "intrinsic_difficulty_counts": {
            "easy": 93,
            "medium": 7,
            "hard": 0
          }
        },
        "counting_probability": {
          "source_sampling_tier_counts": {
            "easy": 30,
            "medium": 40,
            "hard": 30
          },
          "intrinsic_difficulty_counts": {
            "easy": 71,
            "medium": 22,
            "hard": 7
          }
        },
        "number_theory": {
          "source_sampling_tier_counts": {
            "easy": 30,
            "medium": 40,
            "hard": 30
          },
          "intrinsic_difficulty_counts": {
            "easy": 68,
            "medium": 31,
            "hard": 1
          }
        },
        "ratios_percentages": {
          "source_sampling_tier_counts": {
            "easy": 30,
            "medium": 40,
            "hard": 30
          },
          "intrinsic_difficulty_counts": {
            "easy": 86,
            "medium": 14,
            "hard": 0
          }
        }
      },
      "source_sampling_tier_is_intrinsic_difficulty": false
    },
    "pilot": {
      "quota_axis": "source_sampling_tier",
      "n_problems": 50,
      "source_sampling_tier_counts": {
        "easy": 15,
        "medium": 20,
        "hard": 15
      },
      "intrinsic_difficulty_counts": {
        "easy": 38,
        "medium": 11,
        "hard": 1
      },
      "by_domain": {
        "algebra": {
          "source_sampling_tier_counts": {
            "easy": 3,
            "medium": 4,
            "hard": 3
          },
          "intrinsic_difficulty_counts": {
            "easy": 7,
            "medium": 3,
            "hard": 0
          }
        },
        "arithmetic": {
          "source_sampling_tier_counts": {
            "easy": 3,
            "medium": 4,
            "hard": 3
          },
          "intrinsic_difficulty_counts": {
            "easy": 10,
            "medium": 0,
            "hard": 0
          }
        },
        "counting_probability": {
          "source_sampling_tier_counts": {
            "easy": 3,
            "medium": 4,
            "hard": 3
          },
          "intrinsic_difficulty_counts": {
            "easy": 7,
            "medium": 3,
            "hard": 0
          }
        },
        "number_theory": {
          "source_sampling_tier_counts": {
            "easy": 3,
            "medium": 4,
            "hard": 3
          },
          "intrinsic_difficulty_counts": {
            "easy": 5,
            "medium": 4,
            "hard": 1
          }
        },
        "ratios_percentages": {
          "source_sampling_tier_counts": {
            "easy": 3,
            "medium": 4,
            "hard": 3
          },
          "intrinsic_difficulty_counts": {
            "easy": 9,
            "medium": 1,
            "hard": 0
          }
        }
      },
      "source_sampling_tier_is_intrinsic_difficulty": false
    },
    "latency": {
      "quota_axis": "source_sampling_tier",
      "n_problems": 100,
      "source_sampling_tier_counts": {
        "easy": 30,
        "medium": 40,
        "hard": 30
      },
      "intrinsic_difficulty_counts": {
        "easy": 78,
        "medium": 21,
        "hard": 1
      },
      "by_domain": {
        "algebra": {
          "source_sampling_tier_counts": {
            "easy": 6,
            "medium": 8,
            "hard": 6
          },
          "intrinsic_difficulty_counts": {
            "easy": 17,
            "medium": 3,
            "hard": 0
          }
        },
        "arithmetic": {
          "source_sampling_tier_counts": {
            "easy": 6,
            "medium": 8,
            "hard": 6
          },
          "intrinsic_difficulty_counts": {
            "easy": 17,
            "medium": 3,
            "hard": 0
          }
        },
        "counting_probability": {
          "source_sampling_tier_counts": {
            "easy": 6,
            "medium": 8,
            "hard": 6
          },
          "intrinsic_difficulty_counts": {
            "easy": 14,
            "medium": 5,
            "hard": 1
          }
        },
        "number_theory": {
          "source_sampling_tier_counts": {
            "easy": 6,
            "medium": 8,
            "hard": 6
          },
          "intrinsic_difficulty_counts": {
            "easy": 13,
            "medium": 7,
            "hard": 0
          }
        },
        "ratios_percentages": {
          "source_sampling_tier_counts": {
            "easy": 6,
            "medium": 8,
            "hard": 6
          },
          "intrinsic_difficulty_counts": {
            "easy": 17,
            "medium": 3,
            "hard": 0
          }
        }
      },
      "source_sampling_tier_is_intrinsic_difficulty": false
    }
  },
  "n_agent_reviewed": 550,
  "human_reviewed": false,
  "source_sampling_tier_is_intrinsic_difficulty": false,
  "n_source_candidates": 3422,
  "n_unused_source_candidates": 2872,
  "unused_subsets": {
    "source_candidates_not_selected": 2872,
    "pilot_excluded_from_confirmatory_quality": 50,
    "test_not_in_latency": {
      "n_problems": 400,
      "source_sampling_tier_counts": {
        "easy": 120,
        "medium": 160,
        "hard": 120
      },
      "intrinsic_difficulty_counts": {
        "easy": 316,
        "medium": 75,
        "hard": 9
      }
    }
  }
}
```

## Co-Primary Effects

Approximate paired domain-stratified problem-cluster bootstrap: 10000 draws, seed271828.
97.5% percentile CIs use Bonferroni over two co-primary effects; 95% CIs are descriptive.

Quality JFINAL mean3seeds - B13_GREEDY: 10.7333 pp; CI95 [7.866666666666666, 13.666666666666666]; CI97.5 [7.533333333333333, 14.133333333333333].
Latency geometric paired speedup: 1.00192; CI95 [0.8966244257756351, 1.1120552296218766]; CI97.5 [0.883867813624088, 1.1264255001970094].
Latency uses100 test-problem clusters, median9 perarm, all1800 actual records including failures.

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

## Accuracy

| Arm | Cases | Problems | Accuracy % | Descriptive CI95 | Incorrect Cases |
|---|---|---|---|---|---|
| G_SINGLE | 1500 | 500 | 91 | [89.0, 92.86666666666666] | 135 |
| JFINAL | 1500 | 500 | 94.7333 | [93.13333333333331, 96.33333333333331] | 79 |
| B13 | 1500 | 500 | 84.2 | [81.6, 86.73333333333332] | 237 |
| G_GREEDY | 500 | 500 | 92.2 | [89.8, 94.4] | 39 |
| B13_GREEDY | 500 | 500 | 84 | [80.8, 87.2] | 80 |
| Q9_GREEDY | 500 | 500 | 89.2 | [86.4, 91.8] | 54 |
| FIXED_BRANCH0_SHARED | 1500 | 500 | 90.7333 | [88.66666666666666, 92.8] | 139 |
| VOTE4SHARED | 1500 | 500 | 94.6 | [92.93333333333332, 96.2] | 81 |

## Fixed Secondary

Five fixed contrasts; approximate two-sided centered paired stratified cluster bootstrap p-values; Holm5.
These are not exact tests. No McNemar is applied to mean3seed outcomes.

| Method | Baseline | Difference pp | Descriptive CI95 | Approximate p | Holm p |
|---|---|---|---|---|---|
| JFINAL | G_SINGLE | 3.73333 | [2.266666666666667, 5.266666666666667] | 9.999e-05 | 0.00049995 |
| JFINAL | VOTE4SHARED | 0.133333 | [-0.5333333333333333, 0.7999999999999999] | 0.765323 | 0.765323 |
| G_GREEDY | Q9_GREEDY | 3 | [0.2, 5.8] | 0.0439956 | 0.0879912 |
| JFINAL | Q9_GREEDY | 5.53333 | [3.133333333333333, 8.001666666666642] | 9.999e-05 | 0.00049995 |
| G_SINGLE | B13 | 6.8 | [4.531666666666668, 9.133333333333335] | 9.999e-05 | 0.00049995 |

## Shared Controls

Fixed branch0 is chosen before permutation. Plurality uses only valid terminal parses, ties lowest original branch.
Partial/no-candidate sets count incorrect for every control, with no denominator omissions.
Uniform expectation and oracle@4 are gold-informed offline diagnostics, not realizable runtime policies.
No latency is assigned to any shared control.

```json
{
  "n_cases": 1500,
  "planned_candidate_slots": 6000,
  "logged_candidate_slots": 6000,
  "pool_status_counts": {
    "complete": 1500
  },
  "uniform_expected_accuracy_pct_diagnostic": 90.41666666666667,
  "oracle_at4_pct_diagnostic": 96.6,
  "correct_choice_given_any_correct_pct": 98.06763285024155,
  "scope": "Gold-free shared selection; uniform/oracle use gold only offline as diagnostics. No control latency."
}
```

## Audit

Independent strict parser: 6000 predictions, 0 disagreements.

## Limitations

- Agent-reviewed only; human reviewed=false. No human review is claimed.
- Adaptive motivation after prior experiments; historical data and pilots excluded from this test.
- Public benchmark contamination cannot be excluded; domain and difficulty labels are limited proxies.
- Third-party expanded ~13B baseline and distinct training; parameter matching is not compute matching.
- Four samples share generator weights; shared controls condition on the JFINAL pool, not independent generation.
- Different-run latencies are descriptive. Confirmatory latency is conditional on warm same-A10040 execution.
- Reload/loading/warmup excluded from case clocks and reported separately; no invented actual CU.
- Bootstrap CIs and p-values are approximate, not exact coverage or equivalence guarantees.
