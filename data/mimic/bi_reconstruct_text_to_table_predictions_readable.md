# bi_reconstruct_text_to_table_predictions

## idx: 0

### prompt

```text
Reconstruct the table cells masked in the table embedding by using the full markdown content as reference.
Markdown Content is fully observed and does not contain [EMPTY].
The masked positions exist only in the table embedding.
Output JSON list only.
Each element: {"Item": str, "Value": str}

[Table Embedding]
<table>

Markdown Content:
## Patient Demographics [nan]
- Anchor_Age: 20
- Gender: F

## Admissions [2130-06-02 13:37:00]
- Admission_Type: EW EMER.
- Admission_Location: EMERGENCY ROOM
- Admission_Info: None

## Chart Events [2130-06-02 23:36:00]
| Item_Name | Valuenum | Valueuom | Warning |
| ------ | ------ | ------ | ------ |
| 20 Gauge placed in outside facility | 0.0 | nan | 0.0 |
| 20 Gauge Dressing Occlusive | 1.0 | nan | 0.0 |
| 20 Gauge placed in the field | 0.0 | nan | 0.0 |
| Side Rails | nan | nan | 0.0 |
| Safety Measures | nan | nan | 0.0 |
| Safety Measures | nan | nan | 0.0 |
| Safety Measures | nan | nan | 0.0 |
| Safety Measures | nan | nan | 0.0 |
| Safety Measures | nan | nan | 0.0 |
| RUL Lung Sounds | nan | nan | 0.0 |
| RLL Lung Sounds | nan | nan | 0.0 |
| LUL Lung Sounds | nan | nan | 0.0 |
| LLL Lung Sounds | nan | nan | 0.0 |
| Heart Rate | 109.0 | bpm | 0.0 |
| Heart Rhythm | nan | nan | 0.0 |
| Respiratory Rate | 17.0 | insp/min | 0.0 |
| Temperature Fahrenheit | 98.2 | °F | 0.0 |
| Parameters Checked | 1.0 | nan | 0.0 |
| Alarms On | 1.0 | nan | 0.0 |
| Temperature Site | nan | nan | 0.0 |
| ST Segment Monitoring On | 1.0 | nan | 0.0 |
| Nausea and Vomiting (CIWA) | 1.0 | nan | 0.0 |
| Tremor (CIWA) | 3.0 | nan | 0.0 |
| Paroxysmal Sweats | 1.0 | nan | 0.0 |
| Anxiety | 1.0 | nan | 0.0 |
| Auditory Disturbance | 0.0 | nan | 0.0 |
| Agitation | 0.0 | nan | 0.0 |
| Tactile Disturbances | 0.0 | nan | 0.0 |
| Visual Disturbances | 0.0 | nan | 0.0 |
| Headache | 3.0 | nan | 0.0 |
| Orient/Clouding Sensory | 0.0 | nan | 0.0 |
| Insomnia | nan | nan | 0.0 |
| Seizure | nan | nan | 0.0 |
| Sustained Nystagmus | nan | nan | 0.0 |
| Ataxia | nan | nan | 0.0 |
| Slurred Speech | nan | nan | 0.0 |
| Drowsiness | nan | nan | 0.0 |
| Non Invasive Blood Pressure systolic | 135.0 | mmHg | 0.0 |
| Non Invasive Blood Pressure diastolic | 84.0 | mmHg | 0.0 |
| Non Invasive Blood Pressure mean | 93.0 | mmHg | 0.0 |

Output:

```

### target

```json
[{"Item": "Agitation", "Value": "0.0"}, {"Item": "Tremor (CIWA)", "Value": "3.0"}, {"Item": "Paroxysmal Sweats", "Value": "1.0"}]
```

### prediction

```text
{"':20000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000
```

## idx: 1

### prompt

```text
Reconstruct the table cells masked in the table embedding by using the full markdown content as reference.
Markdown Content is fully observed and does not contain [EMPTY].
The masked positions exist only in the table embedding.
Output JSON list only.
Each element: {"Item": str, "Value": str}

[Table Embedding]
<table>

Markdown Content:
## Patient Demographics [nan]
- Anchor_Age: 25
- Gender: F

## Admissions [2157-10-02 14:09:00]
- Admission_Type: OBSERVATION ADMIT
- Admission_Location: WALK-IN/SELF REFERRAL
- Admission_Info: None

## Chart Events [2157-11-26 12:20:00]
| Item_Name | Valuenum | Valueuom | Warning |
| ------ | ------ | ------ | ------ |
| PICC Line Site Appear | nan | nan | 0.0 |
| Multi Lumen Site Appear | nan | nan | 0.0 |
| Arterial line Site Appear | nan | nan | 0.0 |
| Arterial line Waveform Appear | nan | nan | 0.0 |
| Arterial Line placed in outside facility | 0.0 | nan | 0.0 |
| Multi Lumen placed in outside facility | 0.0 | nan | 0.0 |
| PICC Line placed in outside facility | 0.0 | nan | 0.0 |
| 18 Gauge placed in outside facility | 0.0 | nan | 0.0 |
| 18 Gauge Dressing Occlusive | 1.0 | nan | 0.0 |
| Multi Lumen Placement Confirmed by X-ray | nan | nan | 0.0 |
| PICC Line Placement Confirmed by X-ray | nan | nan | 0.0 |
| 18 G Phlebitis Scale | nan | nan | 0.0 |
| 18 G Infiltration Scale | nan | nan | 0.0 |
| 18 Gauge placed in the field | 0.0 | nan | 0.0 |
| PICC Line Power PICC | 0.0 | nan | 0.0 |
| Line Type | nan | nan | 0.0 |
| PICC - Heparin Dependent | 0.0 | nan | 0.0 |
| PICC Biopatch | 1.0 | nan | 0.0 |
| CVL Biopatch | 1.0 | nan | 0.0 |

Output:

```

### target

```json
[{"Item": "PICC Line Power PICC", "Value": "0.0"}]
```

### prediction

```text
{"": ""," "": ""," "": "", "":000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000
```

## idx: 2

### prompt

```text
Reconstruct the table cells masked in the table embedding by using the full markdown content as reference.
Markdown Content is fully observed and does not contain [EMPTY].
The masked positions exist only in the table embedding.
Output JSON list only.
Each element: {"Item": str, "Value": str}

[Table Embedding]
<table>

Markdown Content:
## Patient Demographics [nan]
- Anchor_Age: 67
- Gender: M

## Admissions [2149-12-16 23:36:00]
- Admission_Type: URGENT
- Admission_Location: TRANSFER FROM HOSPITAL
- Admission_Info: None

## Output Events [2149-12-24 04:00:00]
| Item_Name | Value | Valueuom |
| ------ | ------ | ------ |
| TF Residual | 90.0 | mL |
| Fecal Bag | 200.0 | mL |

Output:

```

### target

```json
[{"Item": "TF Residual", "Value": "90.0"}]
```

### prediction

```text
{"":Male"}

{
 "":Male] |":0.0| |

:
] |:0.0|] |:00|] |:0|]|]|]|]|]|]|]|]|]|]|]|]|]|]|]|]|]|]|]|]|]|]|]|]|]|]|]|]|]|]|]|]|]|]|]|]|]|]|]|]|]|]|]|]|]|]|]|]|]|]|]|]|]|]|]|]|]|]|]|]|]|]|]|]|]|]|]|]|]|]|]|]|]|]|]|]|]|]|]|]|]|]|]|]|]|]|]|]|]|]|]|]|]|]|]|]|]|]|]|]|]|]|]|]|]|]|]|]|]|]
```

## idx: 3

### prompt

```text
Reconstruct the table cells masked in the table embedding by using the full markdown content as reference.
Markdown Content is fully observed and does not contain [EMPTY].
The masked positions exist only in the table embedding.
Output JSON list only.
Each element: {"Item": str, "Value": str}

[Table Embedding]
<table>

Markdown Content:
## Patient Demographics [nan]
- Anchor_Age: 76
- Gender: F

## Admissions [2140-11-17 13:42:00]
- Admission_Type: URGENT
- Admission_Location: TRANSFER FROM HOSPITAL
- Admission_Info: None

## Chart Events [2140-11-22 12:00:00]
| Item_Name | Valuenum | Valueuom | Warning |
| ------ | ------ | ------ | ------ |
| Alarms On | 1.0 | nan | 0.0 |
| Parameters Checked | 1.0 | nan | 0.0 |
| Heart Rate | 70.0 | bpm | 0.0 |
| Heart Rhythm | nan | nan | 0.0 |
| Non Invasive Blood Pressure systolic | 133.0 | mmHg | 0.0 |
| Non Invasive Blood Pressure diastolic | 57.0 | mmHg | 0.0 |
| Non Invasive Blood Pressure mean | 74.0 | mmHg | 0.0 |
| Respiratory Rate | 22.0 | insp/min | 0.0 |
| O2 saturation pulseoxymetry | 93.0 | % | 0.0 |
| Ectopy Type 1 | nan | nan | 0.0 |
| Glucose finger stick (range 70-100) | 138.0 | nan | 0.0 |
| Head of Bed | nan | nan | 0.0 |
| Activity | nan | nan | 0.0 |
| Assistance Device | nan | nan | 0.0 |
| Activity Tolerance | nan | nan | 0.0 |
| Therapeutic Bed | nan | nan | 0.0 |
| Pressure Reducing Device | nan | nan | 0.0 |

Output:

```

### target

```json
[{"Item": "Glucose finger stick (range 70-100)", "Value": "138.0"}]
```

### prediction

```text
{"": "O", "":O"]{"O": "",":H", "":", "":000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000
```

## idx: 4

### prompt

```text
Reconstruct the table cells masked in the table embedding by using the full markdown content as reference.
Markdown Content is fully observed and does not contain [EMPTY].
The masked positions exist only in the table embedding.
Output JSON list only.
Each element: {"Item": str, "Value": str}

[Table Embedding]
<table>

Markdown Content:
## Patient Demographics [nan]
- Anchor_Age: 57
- Gender: M

## Admissions [2156-10-22 04:00:00]
- Admission_Type: DIRECT EMER.
- Admission_Location: PHYSICIAN REFERRAL
- Admission_Info: None

## Electronic Medicine Administration Record [2156-11-12 08:56:00]
- Medication: nan
- Event_Txt: Confirmed

Output:

```

### target

```json
[{"Item": "Gender", "Value": "Male"}]
```

### prediction

```text
{"":Gender "", "":M"}

{
 " ": ", ":Male " " ""}

{
 "":Male "":M"}

:
GenderFemale

Activity:S

Di::5

]
[
]
]
]
]
[

]
[]:
]

]
] [] [] [] [] [] [] []
] [] [] [] [] [] [] [] [] [] [] [] [] []

]
] [] [] [] [] [] [] [] [] [] [] []
]
]
]

]
]
]
]
]
]
]
]
]
]
]
]
]
]
]
]
]
]
]
]
]
]
]
]
]
]
]
]
]
]
]
]
]
]
]
]
]
]
]
]
]


]
]
]
]

]
]
]
]
]
]
]
]
]
]
]
]
]
]
]
]
]

]
```

## idx: 5

### prompt

```text
Reconstruct the table cells masked in the table embedding by using the full markdown content as reference.
Markdown Content is fully observed and does not contain [EMPTY].
The masked positions exist only in the table embedding.
Output JSON list only.
Each element: {"Item": str, "Value": str}

[Table Embedding]
<table>

Markdown Content:
## Patient Demographics [nan]
- Anchor_Age: 71
- Gender: M

## Admissions [2130-01-21 18:42:00]
- Admission_Type: EW EMER.
- Admission_Location: EMERGENCY ROOM
- Admission_Info: None

## Ingredient Events [2130-01-24 15:09:00]
| Item_Name | Amount | Amountuom | Rate | Rateuom |
| ------ | ------ | ------ | ------ | ------ |
| Water | 87.02175879478455 | mL | 45.0112533569336 | mL/hour |
| IVF ingredient | 87.02175879478455 | mL | 45.0112533569336 | mL/hour |

Output:

```

### target

```json
[{"Item": "IVF ingredient_Rate", "Value": "45.0112533569336"}]
```

### prediction

```text
{"":M"} Output:{"":M"}

:
:
:
:
:
:
:
:
:
:
:
:
:
:
:
:
:
:
:
:
]
{"":M"}JSON:{"":M"}

:
:
:
]
{"":M"}JSON{"":M"}

:
:
]
: "":A" "":A" "":A]
{"":A}{"A"}]{"":A}{"A"}]{"":A}A
]]:R
{"":A}{"A}]

:
:A]{"":A}{"A}]

:
:
]
{"":A}{"A}]
{" " " " " " " " " " "]{"A}]
{"":A}{"A}]
{"":A}{"A}]
{"":A}{"A}]

:
]
{"":A}{"A}]

]
{"":A}{"A}]

]
{"":A}{"A}]

]
{"":A}{"A}]
{"":A}{"A}]
{"":A}{"A}]
["]
{"":A}{"A}]
["]
{"":A}{"A}]
]]:R]
{"":A}{"A}]

]
```

## idx: 6

### prompt

```text
Reconstruct the table cells masked in the table embedding by using the full markdown content as reference.
Markdown Content is fully observed and does not contain [EMPTY].
The masked positions exist only in the table embedding.
Output JSON list only.
Each element: {"Item": str, "Value": str}

[Table Embedding]
<table>

Markdown Content:
## Patient Demographics [nan]
- Anchor_Age: 81
- Gender: F

## Admissions [2132-07-31 11:50:00]
- Admission_Type: URGENT
- Admission_Location: TRANSFER FROM SKILLED NURSING FACILITY
- Admission_Info: None

## Electronic Medicine Administration Record [2132-07-31 17:57:00]
- Medication: nan
- Event_Txt: Stopped

Output:

```

### target

```json
[{"Item": "Age", "Value": "81"}]
```

### prediction

```text
{"": "", "": ""}

{
 "": "", "":"}

{
}##


The:
ItemKey_Value
___GenderM

JSON[{"":Male","":Female}{"":}]
|":|}{"":|}{"":|}{"":|}
|:|}
|:|}
|:|}
|:|}
|:|}
|:|}
|:|}
|:|}
|:|}
|:|}
|:|}
|:|}
|:|}
|:|}
|:|}
|:|}
|:|}
|:|}
|:|}
|:|}
|:|}
|:|}
|:|}
|:|}
|:|}
|:|}
|:|}
|:|}
|:|}
|:|}
|:|}
|:|}
|:|}
|:|}
|:|}
|:|}
|:|}
|:|}
|:|}
|:|}
|:|}
|:|}
|:|}
|:|}
```

## idx: 7

### prompt

```text
Reconstruct the table cells masked in the table embedding by using the full markdown content as reference.
Markdown Content is fully observed and does not contain [EMPTY].
The masked positions exist only in the table embedding.
Output JSON list only.
Each element: {"Item": str, "Value": str}

[Table Embedding]
<table>

Markdown Content:
## Patient Demographics [nan]
- Anchor_Age: 55
- Gender: M

## Admissions [2186-05-13 20:45:00]
- Admission_Type: OBSERVATION ADMIT
- Admission_Location: PHYSICIAN REFERRAL
- Admission_Info: None

## Electronic Medicine Administration Record [2186-05-19 19:44:00]
| Medication | Event_Txt |
| ------ | ------ |
| Atorvastatin | Administered |
| Heparin | Administered |
| Lidocaine 5% Patch | Not Applied |

Output:

```

### target

```json
[{"Item": "Atorvastatin", "Value": "Atorvastatin"}]
```

### prediction

```text
{"":58 " " " ""}JSON:{"":55 ""}[Sample]:
{"Item "S": "S " " " " " " "}[Ad]{"":55 ""}]{"":55 " " " " " "} " " "]{"S " " " "}[]{"S " " " " " " "}]:S "]|S|S]|S|S]|S|S]|S|S]|S|S]|S|S]|S|S]|S|S]|S|S]|S|S]|S|S]|S|S]|S|S]|S|S]|S|S]|S|S]|S|S]|S|S]|S|S]|S|S]|S|S]|S|S]|S|S]|S|S]|S|S]|S|S]|S|S]|S|S]|S|S]|S|S]|S|S]|S|S]|S|S]|S|S]|
```

## idx: 8

### prompt

```text
Reconstruct the table cells masked in the table embedding by using the full markdown content as reference.
Markdown Content is fully observed and does not contain [EMPTY].
The masked positions exist only in the table embedding.
Output JSON list only.
Each element: {"Item": str, "Value": str}

[Table Embedding]
<table>

Markdown Content:
## Patient Demographics [nan]
- Anchor_Age: 76
- Gender: M

## Admissions [2188-05-18 13:19:00]
- Admission_Type: URGENT
- Admission_Location: TRANSFER FROM HOSPITAL
- Admission_Info: None

## Chart Events [2188-05-26 18:00:00]
| Item_Name | Valuenum | Valueuom | Warning |
| ------ | ------ | ------ | ------ |
| Heart Rate | 86.0 | bpm | 0.0 |
| Arterial Blood Pressure mean | 25.0 | mmHg | 0.0 |
| Non Invasive Blood Pressure systolic | 100.0 | mmHg | 0.0 |
| O2 saturation pulseoxymetry | 97.0 | % | 0.0 |
| Ectopy Type 1 | nan | nan | 0.0 |
| Non Invasive Blood Pressure diastolic | 68.0 | mmHg | 0.0 |
| Non Invasive Blood Pressure mean | 77.0 | mmHg | 0.0 |
| Respiratory Rate | 19.0 | insp/min | 0.0 |
| O2 Delivery Device(s) | nan | nan | 0.0 |
| O2 Flow | 15.0 | L/min | 0.0 |
| Heart Rhythm | nan | nan | 0.0 |

Output:

```

### target

```json
[{"Item": "Non Invasive Blood Pressure diastolic", "Value": "68.0"}]
```

### prediction

```text
{"": "": "": "": "": "": "": "": "": "": "": "": "": "": "": "": "": "": "": "": "": "": "": "": "": "": "": "": "": "": "": "": "": "":":"}{"": "": "": "": "": "": "": "": "": ": "": "": "": ": ":": "": "": "": "": ": "':":"}{"Item_Name Valueu " |0000|Value00000000000|00000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000
```

## idx: 9

### prompt

```text
Reconstruct the table cells masked in the table embedding by using the full markdown content as reference.
Markdown Content is fully observed and does not contain [EMPTY].
The masked positions exist only in the table embedding.
Output JSON list only.
Each element: {"Item": str, "Value": str}

[Table Embedding]
<table>

Markdown Content:
## Patient Demographics [nan]
- Anchor_Age: 74
- Gender: M

## Admissions [2164-07-11 20:48:00]
- Admission_Type: OBSERVATION ADMIT
- Admission_Location: EMERGENCY ROOM
- Admission_Info: None

## Electronic Medicine Administration Record [2164-07-13 15:16:00]
- Medication: Heparin
- Event_Txt: Started

Output:

```

### target

```json
[{"Item": "Age", "Value": "74"}]
```

### prediction

```text
{"":Male "": "", "":"}[] " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " ""}:":]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]
```

## idx: 10

### prompt

```text
Reconstruct the table cells masked in the table embedding by using the full markdown content as reference.
Markdown Content is fully observed and does not contain [EMPTY].
The masked positions exist only in the table embedding.
Output JSON list only.
Each element: {"Item": str, "Value": str}

[Table Embedding]
<table>

Markdown Content:
## Patient Demographics [nan]
- Anchor_Age: 64
- Gender: F

## Admissions [2170-01-14 12:50:00]
- Admission_Type: EW EMER.
- Admission_Location: EMERGENCY ROOM
- Admission_Info: None

## Laboratory Test Events [2170-01-29 02:22:00]
| Item_Name | Valuenum | Valueuom | Ref_Range_Lower | Ref_Range_Upper | Flag | Comments |
| ------ | ------ | ------ | ------ | ------ | ------ | ------ |
| Base Excess | 5.0 | mEq/L | nan | nan | nan | nan |
| Calculated Total CO2 | 35.0 | mEq/L | 21.0 | 30.0 | abnormal | nan |
| Free Calcium | 1.12 | mmol/L | 1.12 | 1.32 | nan | nan |
| Lactate | 0.9 | mmol/L | 0.5 | 2.0 | nan | nan |
| Oxygen Saturation | 94.0 | % | nan | nan | nan | nan |
| pCO2 | 59.0 | mm Hg | 35.0 | 45.0 | abnormal | nan |
| pH | 7.36 | units | 7.35 | 7.45 | nan | nan |
| pO2 | 80.0 | mm Hg | 85.0 | 105.0 | abnormal | nan |
| Specimen Type | nan | nan | nan | nan | nan | ___ |

Output:

```

### target

```json
[{"Item": "Calculated Total CO2", "Value": "35.0"}]
```

### prediction

```text
{"":350,"Value "80000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000
```

## idx: 11

### prompt

```text
Reconstruct the table cells masked in the table embedding by using the full markdown content as reference.
Markdown Content is fully observed and does not contain [EMPTY].
The masked positions exist only in the table embedding.
Output JSON list only.
Each element: {"Item": str, "Value": str}

[Table Embedding]
<table>

Markdown Content:
## Patient Demographics [nan]
- Anchor_Age: 84
- Gender: F

## Admissions [2172-01-02 14:17:00]
- Admission_Type: URGENT
- Admission_Location: TRANSFER FROM HOSPITAL
- Admission_Info: None

## Chart Events [2172-01-06 10:36:00]
| Item_Name | Valuenum | Valueuom | Warning |
| ------ | ------ | ------ | ------ |
| 18 Gauge Site Appear | nan | nan | 0.0 |
| 18 Gauge placed in outside facility | 0.0 | nan | 0.0 |
| 18 Gauge Dressing Occlusive | 1.0 | nan | 0.0 |

Output:

```

### target

```json
[{"Item": "Age", "Value": "84"}]
```

### prediction

```text
{"":18 "":000 "":1.0}{"":18 ":000 ":1.0{"18:ap | |000|000|00|{"18 Site:000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000
```

## idx: 12

### prompt

```text
Reconstruct the table cells masked in the table embedding by using the full markdown content as reference.
Markdown Content is fully observed and does not contain [EMPTY].
The masked positions exist only in the table embedding.
Output JSON list only.
Each element: {"Item": str, "Value": str}

[Table Embedding]
<table>

Markdown Content:
## Patient Demographics [nan]
- Anchor_Age: 72
- Gender: F

## Admissions [2123-03-14 12:54:00]
- Admission_Type: EW EMER.
- Admission_Location: EMERGENCY ROOM
- Admission_Info: None

## Laboratory Test Events [2123-03-28 17:54:00]
| Item_Name | Valuenum | Valueuom | Ref_Range_Lower | Ref_Range_Upper | Flag | Comments |
| ------ | ------ | ------ | ------ | ------ | ------ | ------ |
| Hematocrit | 22.1 | % | 36.0 | 48.0 | abnormal | nan |
| Anion Gap | 14.0 | mEq/L | 8.0 | 20.0 | nan | nan |
| Bicarbonate | 26.0 | mEq/L | 22.0 | 32.0 | nan | nan |
| Calcium, Total | 8.6 | mg/dL | 8.4 | 10.3 | nan | nan |
| Chloride | 107.0 | mEq/L | 96.0 | 108.0 | nan | nan |
| Creatinine | 5.7 | mg/dL | 0.4 | 1.1 | abnormal | nan |
| Glucose | 227.0 | mg/dL | 70.0 | 100.0 | abnormal | IF FASTING, 70-100 NORMAL, >125 PROVISIONAL DIABETES. |
| Magnesium | 2.1 | mg/dL | 1.6 | 2.6 | nan | nan |
| Phosphate | 5.2 | mg/dL | 2.7 | 4.5 | abnormal | nan |
| Potassium | 3.8 | mEq/L | 3.3 | 5.1 | nan | nan |
| Sodium | 143.0 | mEq/L | 133.0 | 145.0 | nan | nan |
| Urea Nitrogen | 80.0 | mg/dL | 6.0 | 20.0 | abnormal | nan |

Output:

```

### target

```json
[{"Item": "Glucose", "Value": "227.0"}, {"Item": "Urea Nitrogen", "Value": "80.0"}]
```

### prediction

```text
{"":nan} "":" " ""]

## Lab Events [21230328175400|] | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
```

## idx: 13

### prompt

```text
Reconstruct the table cells masked in the table embedding by using the full markdown content as reference.
Markdown Content is fully observed and does not contain [EMPTY].
The masked positions exist only in the table embedding.
Output JSON list only.
Each element: {"Item": str, "Value": str}

[Table Embedding]
<table>

Markdown Content:
## Patient Demographics [nan]
- Anchor_Age: 42
- Gender: F

## Admissions [2133-12-07 21:41:00]
- Admission_Type: URGENT
- Admission_Location: TRANSFER FROM HOSPITAL
- Admission_Info: None

## Microbiology Test Events [2133-12-10 13:12:00]
- Test_Name: Blood Culture, Routine
- Dilution_Text: nan
- Interpretation: nan
- Comments: NO GROWTH.  
- Ab_Name: nan

Output:

```

### target

```json
[{"Item": "Age", "Value": "42"}]
```

### prediction

```text
{"":4,"":4,"":44444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444
```

## idx: 14

### prompt

```text
Reconstruct the table cells masked in the table embedding by using the full markdown content as reference.
Markdown Content is fully observed and does not contain [EMPTY].
The masked positions exist only in the table embedding.
Output JSON list only.
Each element: {"Item": str, "Value": str}

[Table Embedding]
<table>

Markdown Content:
## Patient Demographics [nan]
- Anchor_Age: 69
- Gender: M

## Admissions [2112-01-21 20:13:00]
- Admission_Type: EW EMER.
- Admission_Location: EMERGENCY ROOM
- Admission_Info: None

## Chart Events [2112-01-25 19:00:00]
| Item_Name | Valuenum | Valueuom | Warning |
| ------ | ------ | ------ | ------ |
| Minute Volume Alarm - Low | 4.0 | L/min | 0.0 |
| Minute Volume Alarm - High | 24.0 | L/min | 0.0 |
| PEEP set | 8.0 | cmH2O | 0.0 |
| Inspired O2 Fraction | 40.0 | nan | 0.0 |
| ETT Size (ID) | nan | nan | 0.0 |
| ETT Location | nan | nan | 0.0 |
| Ventilator Type | nan | nan | 0.0 |
| Inspired Gas Temp. | 37.0 | °C | 0.0 |
| Paw High | 40.0 | cmH2O | 0.0 |
| Vti High | 850.0 | mL | 0.0 |
| Fspn High | 35.0 | insp/min | 0.0 |
| Apnea Interval | 30.0 | sec | 0.0 |
| RUL Lung Sounds | nan | nan | 0.0 |
| RUL Lung Sounds | nan | nan | 0.0 |
| ETT Mark (cm) | nan | nan | 0.0 |
| Tidal Volume (observed) | 424.0 | mL | 0.0 |
| Tidal Volume (spontaneous) | 424.0 | mL | 0.0 |
| Minute Volume | 7.9 | L/min | 0.0 |
| Respiratory Rate (spontaneous) | 16.0 | insp/min | 0.0 |
| Respiratory Rate (Total) | 16.0 | insp/min | 0.0 |
| Flow Rate (L/min) | 31.7 | L/min | 0.0 |
| Flow Pattern | nan | nan | 0.0 |
| Peak Insp. Pressure | 15.0 | cmH2O | 0.0 |
| Mean Airway Pressure | 9.5 | cmH2O | 0.0 |
| PSV Level | 5.0 | cmH2O | 0.0 |
| ETT Type | nan | nan | 0.0 |
| Flow Sensitivity | nan | nan | 0.0 |
| Slope | nan | nan | 0.0 |
| O2 Delivery Device(s) | nan | nan | 0.0 |
| Humidification | nan | nan | 0.0 |
| Ventilator Tank #1 | 3000.0 | nan | 0.0 |
| Ventilator Tank #2 | 2500.0 | nan | 0.0 |
| Humidifier Water Changed | nan | nan | 0.0 |
| Humidifier Water % Fill Level | nan | nan | 0.0 |
| Ventilator Mode (Hamilton) | 5.0 | nan | 0.0 |
| RLL Lung Sounds | nan | nan | 0.0 |
| LUL Lung Sounds | nan | nan | 0.0 |
| LUL Lung Sounds | nan | nan | 0.0 |
| LLL Lung Sounds | nan | nan | 0.0 |
| Heart Rate | 69.0 | bpm | 0.0 |
| Arterial Blood Pressure systolic | 191.0 | mmHg | 0.0 |
| Arterial Blood Pressure diastolic | 88.0 | mmHg | 0.0 |
| Arterial Blood Pressure mean | 121.0 | mmHg | 0.0 |
| Respiratory Rate | 17.0 | insp/min | 0.0 |
| O2 saturation pulseoxymetry | 99.0 | % | 0.0 |
| Heart Rhythm | nan | nan | 0.0 |
| Ectopy Type 1 | nan | nan | 0.0 |
| Pain Location | nan | nan | 0.0 |
| Pain Level | nan | nan | 0.0 |
| Pain Management | nan | nan | 0.0 |
| Pain Management | nan | nan | 0.0 |
| Pain Level Acceptable | nan | nan | 0.0 |
| Pain Assessment Method | nan | nan | 0.0 |
| Pain Level Response | nan | nan | 0.0 |
| Pain Present | nan | nan | 0.0 |

Output:

```

### target

```json
[{"Item": "Inspired Gas Temp.", "Value": "37.0"}, {"Item": "Respiratory Rate (Total)", "Value": "16.0"}, {"Item": "Minute Volume Alarm - High", "Value": "24.0"}, {"Item": "Arterial Blood Pressure mean", "Value": "121.0"}]
```

### prediction

```text
::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
```

## idx: 15

### prompt

```text
Reconstruct the table cells masked in the table embedding by using the full markdown content as reference.
Markdown Content is fully observed and does not contain [EMPTY].
The masked positions exist only in the table embedding.
Output JSON list only.
Each element: {"Item": str, "Value": str}

[Table Embedding]
<table>

Markdown Content:
## Patient Demographics [nan]
- Anchor_Age: 77
- Gender: M

## Admissions [2125-04-07 20:23:00]
- Admission_Type: OBSERVATION ADMIT
- Admission_Location: WALK-IN/SELF REFERRAL
- Admission_Info: None

## Electronic Medicine Administration Record [2125-05-07 10:28:00]
| Medication | Event_Txt |
| ------ | ------ |
| Acetaminophen IV | Administered |
| Prochlorperazine | Administered |
| Heparin | Administered |
| Bumetanide | Stopped - Unscheduled |

Output:

```

### target

```json
[{"Item": "Acetaminophen IV", "Value": "Acetaminophen IV"}]
```

### prediction

```text
{"":Female "",GenderM}0]|]|]|]|]|]|]|]|]|]|]|]|]|]|]|]|]|]|]|]|]|]|]|]|]|]|]|]|]|]|]|]|]|]|]|]|]|]|]|]|]|]|]|]|]|]|]|]|]|]|]|]|]|]|]|]|]|]|]|]|]|]|]|]|]|]|]|]|]|]|]|||]|]||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
```

## idx: 16

### prompt

```text
Reconstruct the table cells masked in the table embedding by using the full markdown content as reference.
Markdown Content is fully observed and does not contain [EMPTY].
The masked positions exist only in the table embedding.
Output JSON list only.
Each element: {"Item": str, "Value": str}

[Table Embedding]
<table>

Markdown Content:
## Patient Demographics [nan]
- Anchor_Age: 67
- Gender: F

## Admissions [2151-01-10 20:12:00]
- Admission_Type: EW EMER.
- Admission_Location: PHYSICIAN REFERRAL
- Admission_Info: None

## Input Events [2151-01-30 11:23:00]
| Item_Name | Amount | Amountuom | Rate | Rateuom |
| ------ | ------ | ------ | ------ | ------ |
| Norepinephrine | 4.710999488427444 | mg | 0.1379987079417333 | mcg/kg/min |
| NaCl 0.9% | 147.21873199939728 | mL | 52.26700592041016 | mL/hour |

Output:

```

### target

```json
[{"Item": "Norepinephrine_Amount", "Value": "4.710999488427444"}]
```

### prediction

```text
{"":F""}]{"Eq_GUID":1_GUIDAge167|GenderF|Age6|Dr1|Dr_GUID|A|]{"100000000000000000000000000000|00|0|0|0|0]0000000000|]0000000|]0000000|]0000000|]0000000|]0000]0]0]0]0]0]0]0]0]0]0]0]0]0]0]0]0]0]0]0]0]0]0]0]0]0]0]0]0]0]0]0]0]0]0]0]0]0]0]0]0]0]0]0]0]0]0]0]0]0]0]0]0]0]0]0]0]0]0]0]0]0]0]0]0]
```

## idx: 17

### prompt

```text
Reconstruct the table cells masked in the table embedding by using the full markdown content as reference.
Markdown Content is fully observed and does not contain [EMPTY].
The masked positions exist only in the table embedding.
Output JSON list only.
Each element: {"Item": str, "Value": str}

[Table Embedding]
<table>

Markdown Content:
## Patient Demographics [nan]
- Anchor_Age: 68
- Gender: F

## Admissions [2123-09-16 20:08:00]
- Admission_Type: OBSERVATION ADMIT
- Admission_Location: PHYSICIAN REFERRAL
- Admission_Info: None

## Ingredient Events [2123-09-19 20:33:00]
| Item_Name | Amount | Amountuom | Rate | Rateuom |
| ------ | ------ | ------ | ------ | ------ |
| Water | 60.0 | mL | nan | nan |
| Oral/Gastric Ingredient | 60.0 | mL | nan | nan |

Output:

```

### target

```json
[{"Item": "Oral/Gastric Ingredient_Rate", "Value": "Oral/Gastric Ingredient_Rate"}]
```

### prediction

```text
{"":Item "Waterate",Value ""}

:
{"":Item "",Value ""}|]:Item "",Value ""}|]:Item"Rate"Value "|]:{"Gender "M "| | | | ||||||||||||||||||||||||||||||||||||||||||||||||||||||| ||||||||||||||||||||||||||||||||||||||||||||||| |||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||| |
```

## idx: 18

### prompt

```text
Reconstruct the table cells masked in the table embedding by using the full markdown content as reference.
Markdown Content is fully observed and does not contain [EMPTY].
The masked positions exist only in the table embedding.
Output JSON list only.
Each element: {"Item": str, "Value": str}

[Table Embedding]
<table>

Markdown Content:
## Patient Demographics [nan]
- Anchor_Age: 20
- Gender: M

## Vitalsign [2169-03-02 20:14:00]
- Temperature: 95.5
- Heartrate: 103.0
- Resprate: 16.0
- O2Sat: 96.0
- Sbp: 106.0
- Dbp: 65.0
- Pain: 2
- Rhythm: nan

Output:

```

### target

```json
[{"Item": "Temperature", "Value": "95.5"}]
```

### prediction

```text
{":20"-:20|-:20|-:20|-:20|

---
V []:210--1:1:0:
Temperature .5-
Hear:1000
ate:1000-
ate:1000
ate:1000
ate:1000
ate:1000-ate:1000
ate:1000-ate:1000-:1000-:1000-:1000-:1000-:1000-:1000-:1000-:1000-:1000-:1000-:1000-:1000-:1000-:1000-:1000-:1000-:1000-:1000-:1000-:1000-:1000-:1000-:1000-:100
```

## idx: 19

### prompt

```text
Reconstruct the table cells masked in the table embedding by using the full markdown content as reference.
Markdown Content is fully observed and does not contain [EMPTY].
The masked positions exist only in the table embedding.
Output JSON list only.
Each element: {"Item": str, "Value": str}

[Table Embedding]
<table>

Markdown Content:
## Patient Demographics [nan]
- Anchor_Age: 56
- Gender: M

## Admissions [2162-07-30 05:26:00]
- Admission_Type: OBSERVATION ADMIT
- Admission_Location: EMERGENCY ROOM
- Admission_Info: None

## Ingredient Events [2162-07-30 08:05:00]
| Item_Name | Amount | Amountuom | Rate | Rateuom |
| ------ | ------ | ------ | ------ | ------ |
| Water | 11.938107192516329 | mL | 42.13449478149414 | mL/hour |
| Calories | 2.029478318989277 | Kcal | 7.162864685058594 | mL/hour |
| IVF ingredient | 11.938107192516329 | mL | 42.13449478149414 | mL/hour |

Output:

```

### target

```json
[{"Item": "Age", "Value": "56"}]
```

### prediction

```text
{"":A "",A "|L "":"/ "("|R":"/"}:R"}("|":"/|("|":"/"}L3|3|3|3|333330000|3|3|3|3|3|3|3|3|3|3|3|3|3|3|3|3|3|3|3|3|3|3|3|3|3RR|3RR|3RR|3RR|3RR|3RR|3RR"}:A|3|3|3|3|3|3Output "":A_R|3|3|3|3|3("|3|("|3|("|3|("|3|("|3|("|3|("|3|("|3|("|3|("|3|{"3|{"3|{"3|333333333333333333333333333333333333333333333333333333333333333333333333333
```
