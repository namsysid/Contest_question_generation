from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from src.circuit_lab.model_client import embed_texts, generate_json

from .common import read_jsonl, write_jsonl
from .pipeline import BATCH_AUDIT_SYSTEM, create_item, make_plan, normalized, retrieve_exemplars


REJECT_IDS = {
    "machines-b-generated-002", "machines-b-generated-003", "machines-b-generated-005",
    "machines-b-generated-008", "machines-b-generated-011", "machines-b-generated-015",
    "machines-b-generated-024", "machines-b-generated-026", "machines-b-generated-028",
    "machines-b-generated-009", "machines-b-generated-014", "machines-b-generated-018",
    "machines-b-generated-022",
    "machines-b-generated-012", "machines-b-generated-017", "machines-b-generated-021",
    "machines-b-generated-101", "machines-b-generated-109", "machines-b-generated-111",
    "machines-b-generated-119", "machines-b-generated-132",
}
REGENERATE_IDS: set[str] = set()

CHALLENGES = [
    ("levers_torque", 3, "numeric", "Interpret a text data table from repeated lever trials to infer an unknown load using torque relationships."),
    ("pulleys", 3, "multiple_choice", "Compare measured effort forces for two pulley arrangements and diagnose which has the greater fractional loss."),
    ("gears_compound", 3, "short_answer", "Diagnose a slipping or misconnected stage from input, intermediate, and output speeds in a compound gear train."),
    ("inclined_planes_wedges", 3, "multiple_choice", "Choose between ramp designs under both a maximum-force and a maximum-distance constraint."),
    ("wheel_axle_screws", 3, "numeric", "Analyze a wheel-and-axle driving a screw jack, including efficiency and required output force."),
    ("levers_torque", 3, "numeric", "Analyze a lever that pushes a load up an inclined plane; combine torque balance and ramp mechanical advantage."),
    ("gears_compound", 3, "numeric", "Analyze a gear train driving a pulley drum; combine gear speed ratio with rope lifting speed or time."),
    ("mechanical_advantage", 3, "multiple_choice", "Select a design satisfying both a limited effort force and limited effort distance using the force-distance tradeoff."),
    ("efficiency_work_power", 2, "short_answer", "Explain from input/output work data why energy is not destroyed and identify two realistic loss mechanisms."),
    ("levers_torque", 2, "multiple_choice", "Use repeated measurements to identify an anomalous trial and the scientifically appropriate response in a Machines experiment."),
    ("efficiency_work_power", 3, "numeric", "Combine load weight, vertical rise, elapsed time, and efficiency to determine input power for a compound lift."),
    ("inclined_planes_wedges", 3, "numeric", "Analyze a real wedge using length, thickness, efficiency, and required splitting force; determine the minimum effort."),
    ("levers_torque", 2, "multiple_choice", "Apply Archimedes' lever principle to choose a fulcrum change and identify the force-distance tradeoff; do not ask pure trivia."),
    ("inclined_planes_wedges", 3, "numeric", "Determine the efficiency of a rough inclined plane from its angle and kinetic-friction coefficient."),
    ("levers_torque", 3, "numeric", "Solve static equilibrium for a loaded lever whose own weight contributes a second torque."),
    ("pulleys", 4, "short_answer", "Derive the IMA of a Weston differential pulley from the unequal upper-pulley radii, then evaluate it."),
    ("pulleys", 3, "numeric", "Combine differential-pulley geometry, efficiency, chain speed, and power to determine useful output power."),
    ("wheel_axle_screws", 3, "numeric", "Use thread pitch and a supplied thread-depth approximation to determine a screw's minor diameter."),
    ("levers_torque", 4, "numeric", "Use force and torque equilibrium to find the minimum ground friction coefficient for a ladder against a smooth wall."),
    ("wheel_axle_screws", 5, "short_answer", "Analyze a composite flywheel's rotational inertia and angular acceleration with axle friction."),
    ("mechanical_advantage", 3, "short_answer", "Combine lever and movable-pulley stages and distinguish geometric distance ratio from efficiency losses."),
    ("efficiency_work_power", 3, "multiple_choice", "Use repeated force measurements from a ramp experiment to estimate efficiency and handle measurement scatter."),
    ("gears_compound", 2, "multiple_choice", "Predict the simultaneous output-speed and torque changes when a driven gear is replaced."),
    ("inclined_planes_wedges", 3, "numeric", "Determine real wedge efficiency by comparing its actual mechanical advantage with angle-based IMA."),
    ("wheel_axle_screws", 3, "numeric", "Use wheel-and-axle velocity ratio and efficiency to determine the maximum steady load."),
    ("levers_torque", 1, "multiple_choice", "Recognize how a single lever-dimension change affects IMA; keep the stem concise."),
    ("pulleys", 1, "multiple_choice", "Distinguish the force and direction functions of one fixed pulley and one movable pulley."),
    ("inclined_planes_wedges", 1, "numeric", "Calculate an inclined plane's IMA directly from its length and vertical rise."),
    ("wheel_axle_screws", 2, "numeric", "Use measured wheel-and-axle forces to calculate AMA and efficiency from IMA."),
    ("wheel_axle_screws", 2, "numeric", "Relate screw pitch and number of turns to linear advance, including a fractional turn."),
    ("gears_compound", 2, "multiple_choice", "Predict both speed and ideal torque changes for a single gear reduction."),
    ("levers_torque", 4, "short_answer", "Analyze a sliding-counterweight balance including beam weight, then determine its sensitivity to a changed load."),
    ("levers_torque", 1, "multiple_choice", "Identify the class of a familiar lever from the positions of fulcrum, load, and effort."),
    ("efficiency_work_power", 1, "multiple_choice", "Identify a concrete machine change that reduces efficiency by increasing dissipative losses."),
    ("mechanical_advantage", 1, "multiple_choice", "Identify the simple machines combined in a familiar compound machine."),
    ("efficiency_work_power", 1, "multiple_choice", "Apply ideal work conservation qualitatively to a longer inclined plane."),
    ("wheel_axle_screws", 2, "short_answer", "Compare single-start and double-start screws with the same thread pitch for advance per turn and IMA."),
    ("pulleys", 3, "multiple_choice", "Interpret a text data table of loads and efforts to determine how pulley efficiency changes with load."),
    ("mechanical_advantage", 2, "multiple_choice", "Choose a machine modification when effort force is limited but extra effort distance is acceptable."),
    ("gears_compound", 3, "numeric", "Combine a gear reduction, rope-drum torque, and overall efficiency to find required motor torque."),
    ("wheel_axle_screws", 1, "multiple_choice", "Predict how increasing wheel radius changes ideal mechanical advantage when axle radius is fixed."),
    ("inclined_planes_wedges", 1, "multiple_choice", "Compare wedge shapes qualitatively using length-to-thickness mechanical advantage."),
    ("levers_torque", 1, "multiple_choice", "Identify a third-class lever from the order of fulcrum, effort, and load in a familiar tool."),
    ("mechanical_advantage", 5, "short_answer", "Derive and apply the force and distance limits of a lever-driven differential hoist with real efficiency."),
    ("mechanical_advantage", 4, "multiple_choice", "Select the only machine design satisfying simultaneous effort-force and effort-travel limits using IMA and efficiency."),
    ("inclined_planes_wedges", 3, "multiple_choice", "Infer static friction from a critical ramp angle, then predict the acceleration after sliding begins."),
    ("levers_torque", 3, "multiple_choice", "Identify the complete force-and-torque conditions required for static equilibrium of a machine component."),
    ("levers_torque", 1, "multiple_choice", "Choose the force angle that produces maximum torque on a wrench."),
    ("levers_torque", 4, "short_answer", "Solve a horizontal drawbridge's cable tension and two hinge-reaction components using torque and force equilibrium."),
    ("mechanical_advantage", 1, "numeric", "Calculate actual mechanical advantage directly from measured load and effort forces."),
    ("efficiency_work_power", 1, "numeric", "Calculate machine efficiency directly from measured input and useful output work."),
    ("mechanical_advantage", 1, "numeric", "Calculate total IMA for two ideal machine stages connected in series."),
    ("wheel_axle_screws", 3, "multiple_choice", "Determine whether a screw is self-locking from lead angle and friction, then predict the effect of lubrication."),
    ("efficiency_work_power", 1, "numeric", "Calculate useful output power from a machine's load, lift distance, and elapsed time."),
    ("pulleys", 4, "short_answer", "State the Weston differential-hoist IMA and apply real efficiency to find effort, concisely."),
    ("efficiency_work_power", 1, "multiple_choice", "Predict how added fulcrum friction changes IMA, AMA, and efficiency for an unchanged lever."),
    ("wheel_axle_screws", 1, "multiple_choice", "Predict both force and distance consequences when axle radius doubles in an ideal wheel-and-axle."),
    ("levers_torque", 1, "multiple_choice", "Identify a different wrench force-and-length combination that produces the same torque."),
    ("wheel_axle_screws", 1, "multiple_choice", "Compare screw pitch effects on advance per turn and ideal force advantage."),
    ("pulleys", 2, "multiple_choice", "Compare absolute energy loss and efficiency for two measured pulley lifts."),
    ("efficiency_work_power", 1, "multiple_choice", "Compare useful work and output power when the same winch lift is completed in half the time."),
]

VETTED_PLANS: dict[int, dict[str, Any]] = {
    1: {"target_skill": "compare fractional losses in two pulley systems", "novel_context": "two measured pulley rigs",
        "fixed_givens": ["Rig A: IMA 4, load 480 N, effort 150 N", "Rig B: IMA 6, load 720 N, effort 160 N"],
        "verification": {"expected_answer": "Rig B; A is 80% efficient (20% loss), B is 75% efficient (25% loss)",
                         "calculation": "A: AMA=480/150=3.2, efficiency=3.2/4=0.80. B: AMA=720/160=4.5, efficiency=4.5/6=0.75."}},
    2: {"target_skill": "diagnose a faulty stage using compound gear speed data", "novel_context": "measured mixer gear train",
        "fixed_givens": ["A=12 teeth drives B=36", "B and C=10 teeth share a shaft", "C drives D=40 teeth",
                         "A=600 rpm; measured B=200 rpm, C=200 rpm, D=80 rpm"],
        "verification": {"expected_answer": "The C-D stage is slipping or misconnected; D should turn at 50 rpm",
                         "calculation": "B=C=600(12/36)=200 rpm; ideal D=200(10/40)=50 rpm, not 80 rpm."}},
    3: {"target_skill": "select a ramp under force and distance constraints", "novel_context": "loading-dock ramp selection",
        "fixed_givens": ["600 N load raised 1.2 m", "Ramp A length 6.0 m", "Ramp B length 4.0 m",
                         "both 80% efficient", "effort may not exceed 150 N", "effort distance may not exceed 5.0 m"],
        "verification": {"expected_answer": "Neither ramp", "calculation": "A ideal force=120 N and actual=150 N but distance 6 m fails. B ideal force=180 N and actual=225 N, so force fails."}},
    4: {"target_skill": "combine screw IMA with efficiency", "novel_context": "handwheel screw press",
        "fixed_givens": ["handwheel radius 0.20 m", "screw pitch 0.0040 m", "efficiency 30%", "effort 40 N", "use pi=3.14"],
        "verification": {"expected_answer": "3768 N", "expected_value": 3768,
                         "expression": "0.30*(2*3.14*0.20/0.0040)*40",
                         "calculation": "IMA=2πr/p=314; AMA=0.30(314)=94.2; output=94.2(40)=3768 N."}},
    5: {"target_skill": "combine lever and inclined-plane mechanical advantage", "novel_context": "lever-operated loading ramp",
        "fixed_givens": ["lever effort arm 1.2 m and load arm 0.30 m", "ramp length 5.0 m and height 1.0 m",
                         "lever output drives ramp load", "1000 N load", "ideal system"],
        "verification": {"expected_answer": "50 N", "expected_value": 50,
                         "expression": "1000/((1.2/0.30)*(5.0/1.0))",
                         "calculation": "Lever IMA=4 and ramp IMA=5, so total IMA=20. Effort=1000/20=50 N."}},
    6: {"target_skill": "combine compound gear ratio with drum lifting speed", "novel_context": "gear-driven rope drum",
        "fixed_givens": ["A=12 teeth drives B=48", "B shares shaft with C=15 teeth", "C drives D=45 teeth",
                         "A turns 240 rpm", "D drives a drum of radius 0.060 m", "lift distance 3.0 m", "use pi=3.14"],
        "verification": {"expected_answer": "23.9 s", "expected_value": 23.8853503185,
                         "expression": "3.0/((240*(12/48)*(15/45))*(2*3.14*0.060)/60)",
                         "calculation": "D=20 rpm. Rope speed=20(2π·0.060)/60=0.1256 m/s. Time=3.0/0.1256=23.9 s."}},
    7: {"target_skill": "satisfy simultaneous force and distance constraints", "novel_context": "select among ideal lifting designs",
        "fixed_givens": ["600 N load raised 1.0 m", "effort must be at most 130 N", "effort distance at most 5.0 m",
                         "design IMAs are A=4, B=6, C=3, D=5", "all designs ideal"],
        "verification": {"expected_answer": "Design D", "calculation": "A needs 150 N; B needs 6 m; C needs 200 N; D needs 120 N over 5 m and alone satisfies both."}},
    8: {"target_skill": "interpret energy accounting in a real machine", "novel_context": "measured work audit",
        "fixed_givens": ["input work 800 J", "useful output work 600 J"],
        "verification": {"expected_answer": "200 J became non-useful energy such as thermal energy, sound, or deformation; total energy is conserved",
                         "calculation": "800-600=200 J is transferred to non-useful forms; it is not destroyed."}},
    9: {"target_skill": "identify and respond to anomalous machine measurements", "novel_context": "repeated lever-effort experiment",
        "fixed_givens": ["same lever geometry and load for every trial", "measured efforts: 49 N, 51 N, 50 N, 81 N"],
        "verification": {"expected_answer": "Trial 4 is anomalous; inspect/reset the apparatus and repeat it rather than deleting it without investigation",
                         "calculation": "The first three cluster near 50 N; 81 N is isolated under nominally identical conditions."}},
    10: {"target_skill": "combine lifting work, time, and efficiency to find input power", "novel_context": "compound platform lift",
         "fixed_givens": ["load weight 800 N", "vertical rise 2.5 m", "time 20 s", "overall efficiency 62.5%"],
         "verification": {"expected_answer": "160 W", "expected_value": 160,
                          "expression": "(800*2.5/20)/0.625",
                          "calculation": "Useful work=2000 J; output power=100 W; input power=100/0.625=160 W."}},
    11: {"target_skill": "combine wedge IMA and efficiency to meet an output-force requirement", "novel_context": "wood-splitting wedge selection",
         "fixed_givens": ["wedge length 0.20 m", "wedge thickness 0.040 m", "efficiency 60%", "required splitting force 1000 N"],
         "verification": {"expected_answer": "333.3 N", "expected_value": 333.3333333333,
                          "expression": "1000/(0.60*(0.20/0.040))",
                          "calculation": "IMA=0.20/0.040=5; AMA=0.60(5)=3; minimum effort=1000/3=333.3 N."}},
    12: {"target_skill": "apply lever geometry and the force-distance tradeoff", "novel_context": "Archimedes-inspired stone-moving lever",
         "fixed_givens": ["first-class lever", "same effort force", "move fulcrum closer to load", "ideal work is conserved"],
         "verification": {"expected_answer": "Move the fulcrum closer to the load; required effort force decreases but the effort point moves farther",
                          "calculation": "A shorter load arm and longer effort arm increase IMA. Ideal work conservation trades reduced effort force for greater effort distance."}},
    13: {"target_skill": "combine ramp force components and kinetic friction to determine efficiency", "novel_context": "rough loading ramp",
         "fixed_givens": ["crate weight 600 N", "ramp angle 30 degrees", "coefficient of kinetic friction 0.25",
                           "crate moves uphill at constant speed", "effort is parallel to ramp", "use cos(30 degrees)=0.866"],
         "verification": {"expected_answer": "69.8%", "expected_value": 69.783,
                          "expression": "(600*0.5)/(600*0.5+0.25*600*0.866)*100",
                          "calculation": "Down-slope gravity is 300 N and friction is 0.25(600 cos30)=129.9 N, so effort is 429.9 N. For equal ramp distances, efficiency is ideal effort/actual effort = 300/429.9 = 69.8%."}},
    14: {"target_skill": "balance three torques including the lever's own weight", "novel_context": "loaded maintenance beam",
         "fixed_givens": ["horizontal 4.0 m uniform beam", "pivot 1.5 m from left end", "beam weight 120 N acts at its center",
                           "300 N load hangs 0.5 m from left end", "downward effort is applied 3.5 m from left end"],
         "verification": {"expected_answer": "120 N", "expected_value": 120,
                          "expression": "(300*(1.5-0.5)-120*(2.0-1.5))/(3.5-1.5)",
                          "calculation": "About the pivot, the load supplies 300 N m counterclockwise. Beam weight supplies 60 N m clockwise. The effort arm is 2.0 m, so 300=60+2F and F=120 N."}},
    15: {"target_skill": "derive differential-pulley IMA from chain travel", "novel_context": "Weston differential chain hoist",
         "fixed_givens": ["joined upper sheaves have radii R=18 cm and r=14 cm", "one lower movable sheave",
                           "one turn pulls 2 pi R of chain while releasing 2 pi r", "the lower block is supported by two chain portions"],
         "verification": {"expected_answer": "IMA = 2R/(R-r) = 9",
                          "calculation": "One turn changes total supporting-chain length by 2π(R-r), so the two supported portions shorten by half that amount and the load rises π(R-r). Effort travels 2πR. Thus IMA=2πR/[π(R-r)]=2R/(R-r)=36/4=9."}},
    16: {"target_skill": "connect differential-pulley velocity ratio, efficiency, force, and useful power", "novel_context": "instrumented differential hoist",
         "fixed_givens": ["large and small joined sheave radii are 16 cm and 12 cm", "IMA=2R/(R-r)",
                           "efficiency 75%", "effort force 120 N", "effort chain speed 0.40 m/s"],
         "verification": {"expected_answer": "36 W", "expected_value": 36,
                          "expression": "0.75*120*0.40",
                          "calculation": "IMA=2(16)/(16-12)=8, so load speed is 0.40/8=0.050 m/s. AMA=0.75(8)=6, so load=720 N. Useful power=720(0.050)=36 W (also 0.75 times the 48 W input)."}},
    17: {"target_skill": "apply screw-thread geometry to minor diameter", "novel_context": "replacement vise screw",
         "fixed_givens": ["major diameter 14.0 mm", "pitch 1.25 mm", "for this 60-degree thread use depth per side h=0.65 times pitch",
                           "minor diameter equals major diameter minus twice the thread depth"],
         "verification": {"expected_answer": "12.38 mm (about 12.4 mm)", "expected_value": 12.375,
                          "expression": "14.0-2*0.65*1.25",
                          "calculation": "Thread depth per side is 0.65(1.25)=0.8125 mm. Subtracting two depths gives 14.0-1.625=12.375 mm, or 12.38 mm."}},
    18: {"target_skill": "combine ladder force balance and torque balance", "novel_context": "uniform ladder against a frictionless wall",
         "fixed_givens": ["uniform 5.0 m ladder weighs 300 N", "ladder makes 60 degrees with level ground", "wall is smooth",
                           "ladder is just on the verge of slipping", "use sin60=0.866 and cos60=0.500"],
         "verification": {"expected_answer": "0.289", "expected_value": 0.288684,
                          "expression": "0.5*(0.500/0.866)",
                          "calculation": "Torque about the foot gives N_wall(5 sin60)=300(2.5 cos60), so N_wall=86.6 N. Horizontal balance gives friction=86.6 N and vertical balance gives N_ground=300 N. Therefore μ_min=86.6/300=0.289."}},
    19: {"target_skill": "combine moments of inertia, net torque, and angular acceleration", "novel_context": "composite demonstration flywheel",
         "fixed_givens": ["solid disk mass 4.0 kg and radius 0.30 m", "thin ring mass 2.0 kg and radius 0.40 m, concentric with disk",
                           "tangential force 10 N at radius 0.40 m", "axle friction supplies opposing torque 0.50 N m",
                           "I_disk=(1/2)MR^2, I_ring=MR^2, and alpha=net torque divided by total I"],
         "verification": {"expected_answer": "7.0 rad/s^2", "expected_value": 7,
                          "expression": "(10*0.40-0.50)/(0.5*4.0*0.30**2+2.0*0.40**2)",
                          "calculation": "I_total=0.5(4)(0.30^2)+2(0.40^2)=0.18+0.32=0.50 kg m^2. Net torque=4.0-0.50=3.5 N m. Thus alpha=3.5/0.50=7.0 rad/s^2."}},
    20: {"target_skill": "separate compound-machine force advantage, distance ratio, and efficiency", "novel_context": "lever-operated movable-pulley lift",
         "fixed_givens": ["lever effort arm 1.0 m and load arm 0.25 m", "lever output pulls the free end of a single movable pulley with two supporting strands",
                           "overall efficiency 60%", "load weight 900 N", "load rises 0.40 m"],
         "verification": {"expected_answer": "187.5 N effort and 3.2 m effort travel",
                          "calculation": "Geometric IMA is (1.0/0.25)(2)=8. AMA=0.60(8)=4.8, so effort=900/4.8=187.5 N. Geometry fixes the ideal distance ratio even with losses, so effort travel=8(0.40)=3.2 m."}},
    21: {"target_skill": "estimate machine efficiency from repeated experimental force data", "novel_context": "three-trial ramp experiment",
         "fixed_givens": ["ramp length 3.0 m and height 0.75 m", "crate weight 400 N", "measured constant-speed efforts are 125 N, 130 N, and 127 N",
                           "use the mean measured effort", "efficiency equals ideal effort divided by actual effort"],
         "verification": {"expected_answer": "about 78.5%, using the mean rather than selecting one trial",
                          "calculation": "Ideal effort=400(0.75/3.0)=100 N. Mean actual effort=(125+130+127)/3=127.33 N. Efficiency=100/127.33=0.785 or 78.5%."}},
    22: {"target_skill": "predict the speed-torque tradeoff of a gear substitution", "novel_context": "winch gearbox modification",
         "fixed_givens": ["20-tooth driver initially meshes with 60-tooth driven gear", "driven gear is replaced by a 40-tooth gear",
                           "motor speed and ideal input torque stay fixed", "gears are ideal"],
         "verification": {"expected_answer": "output speed increases while output torque decreases",
                          "calculation": "Changing the driven ratio from 60/20=3 to 40/20=2 reduces the reduction. Output speed changes from one-third to one-half of motor speed, while ideal torque multiplication falls from 3 to 2."}},
    23: {"target_skill": "compare actual and ideal wedge mechanical advantage", "novel_context": "instrumented symmetric lifting wedge",
         "fixed_givens": ["each wedge face is 12 degrees from the direction of the applied effort", "use IMA=1/tan(theta) and tan12=0.213",
                           "wedge lifts with total output force 2500 N", "measured input force 700 N"],
         "verification": {"expected_answer": "76.1%", "expected_value": 76.071,
                          "expression": "(2500/700)/(1/0.213)*100",
                          "calculation": "AMA=2500/700=3.571. IMA=1/0.213=4.695. Efficiency=AMA/IMA=3.571/4.695=0.7607, or 76.1%."}},
    24: {"target_skill": "combine wheel-and-axle geometry with efficiency", "novel_context": "hand-cranked well windlass",
         "fixed_givens": ["crank radius 0.25 m", "rope winds on axle radius 0.050 m", "tangential effort 80 N",
                           "efficiency 70%", "load rises at constant speed"],
         "verification": {"expected_answer": "280 N", "expected_value": 280,
                          "expression": "0.70*(0.25/0.050)*80",
                          "calculation": "IMA=0.25/0.050=5. Effective AMA=0.70(5)=3.5. The maximum steady load is 3.5(80)=280 N."}},
    25: {"target_skill": "recognize the lever-arm ratio that controls IMA", "novel_context": "adjustable pry bar",
         "fixed_givens": ["effort arm initially 0.40 m", "load arm remains 0.20 m", "effort arm is increased to 0.80 m", "ideal lever"],
         "verification": {"expected_answer": "The IMA doubles, from 2 to 4",
                          "calculation": "Lever IMA=effort arm/load arm. It changes from 0.40/0.20=2 to 0.80/0.20=4."}},
    26: {"target_skill": "distinguish fixed- and movable-pulley functions", "novel_context": "two basic flag-and-load pulley arrangements",
         "fixed_givens": ["Arrangement A is one fixed pulley attached to a beam", "Arrangement B is one movable pulley attached to the load",
                           "ideal massless rope and frictionless pulleys", "the movable pulley has two supporting rope strands"],
         "verification": {"expected_answer": "A changes the pull direction with IMA 1; B has IMA 2",
                          "calculation": "A fixed pulley redirects the force but has one effective supporting strand. The movable load is supported by two equal-tension strands, so its IMA is 2."}},
    27: {"target_skill": "calculate inclined-plane IMA from geometry", "novel_context": "equipment loading ramp",
         "fixed_givens": ["ramp surface length 4.8 m", "vertical rise 1.2 m", "ideal ramp"],
         "verification": {"expected_answer": "4", "expected_value": 4,
                          "expression": "4.8/1.2", "calculation": "For an inclined plane, IMA=length/height=4.8/1.2=4."}},
    28: {"target_skill": "calculate wheel-and-axle efficiency from geometry and force measurements", "novel_context": "tested hand winch",
         "fixed_givens": ["wheel radius 0.24 m", "axle radius 0.040 m", "effort force 70 N", "steady load 336 N"],
         "verification": {"expected_answer": "80%", "expected_value": 80,
                          "expression": "((336/70)/(0.24/0.040))*100",
                          "calculation": "IMA=0.24/0.040=6. AMA=336/70=4.8. Efficiency=AMA/IMA=4.8/6=0.80, or 80%."}},
    29: {"target_skill": "relate screw pitch, turns, and axial travel", "novel_context": "vise jaw adjustment",
         "fixed_givens": ["single-start screw advances 2.0 mm per complete turn", "jaw must advance 15.0 mm", "partial turns are allowed"],
         "verification": {"expected_answer": "7.5 turns", "expected_value": 7.5,
                          "expression": "15.0/2.0", "calculation": "Turns=required advance/pitch=15.0 mm/(2.0 mm per turn)=7.5 turns."}},
    30: {"target_skill": "connect gear reduction to both speed and torque", "novel_context": "slow-speed mixer gearbox",
         "fixed_givens": ["18-tooth motor gear drives a 54-tooth output gear", "ideal gears", "motor speed and torque are fixed"],
         "verification": {"expected_answer": "The output turns at one-third the motor speed with three times the motor torque",
                          "calculation": "The driven-to-driver tooth ratio is 54/18=3. Speed is divided by 3; ideal torque is multiplied by 3."}},
    31: {"target_skill": "use repeated torque balances to find counterweight position and sensitivity", "novel_context": "sliding-counterweight platform scale",
         "fixed_givens": ["400 N load acts 0.30 m left of pivot", "uniform beam weight 100 N acts 0.20 m right of pivot",
                           "80 N counterweight slides to the right", "then the load increases to 440 N", "beam remains horizontal in each balance"],
         "verification": {"expected_answer": "1.25 m initially; 1.40 m after the load change, so move it 0.15 m farther right",
                          "calculation": "Initial balance: 400(0.30)=100(0.20)+80x, so x=1.25 m. New balance: 440(0.30)=20+80x, so x=1.40 m. The required shift is 0.15 m."}},
    32: {"target_skill": "classify a lever from component order", "novel_context": "loaded wheelbarrow",
         "fixed_givens": ["wheel axle is the fulcrum", "soil in the tray is the load", "person lifts at the handles", "the load lies between fulcrum and effort"],
         "verification": {"expected_answer": "second-class lever",
                          "calculation": "A second-class lever places the load between the fulcrum and the effort, as in this wheelbarrow."}},
    33: {"target_skill": "identify a cause of reduced efficiency", "novel_context": "aging block-and-tackle",
         "fixed_givens": ["same pulley geometry and load before and after maintenance", "axle bearings become rough", "more effort force is then required"],
         "verification": {"expected_answer": "increased axle friction reduces efficiency",
                          "calculation": "Geometry leaves IMA unchanged, but rough bearings dissipate more input energy as heat, increasing required effort and reducing AMA/IMA."}},
    34: {"target_skill": "identify simple machines within a compound device", "novel_context": "ordinary scissors",
         "fixed_givens": ["scissors rotate about a central pivot", "handles apply effort about the pivot", "tapered cutting edges force material apart"],
         "verification": {"expected_answer": "two first-class levers with wedge-shaped blades",
                          "calculation": "Each half rotates as a first-class lever about the pivot, and each sharpened blade acts as a wedge at the material."}},
    35: {"target_skill": "apply ideal force-distance-work tradeoff", "novel_context": "choosing between two frictionless ramps",
         "fixed_givens": ["both ramps raise the same load to the same height", "Ramp B is longer than Ramp A", "both are ideal and frictionless"],
         "verification": {"expected_answer": "Ramp B needs less force over more distance, while input work stays the same",
                          "calculation": "For ideal machines, input work equals the same gain in gravitational energy. A longer ramp increases IMA, trading reduced effort force for increased effort distance."}},
    36: {"target_skill": "compare lead and IMA of single-start and double-start screws", "novel_context": "two otherwise identical screw jacks",
         "fixed_givens": ["both screws have 2.0 mm pitch and the same handle radius", "Screw A is single-start", "Screw B is double-start",
                           "lead equals pitch times number of starts", "screw IMA is handle circumference divided by lead"],
         "verification": {"expected_answer": "B advances 4.0 mm per turn versus A's 2.0 mm, so B has half A's IMA and moves faster but needs more ideal effort",
                          "calculation": "A lead is 2.0 mm; B lead is 2(2.0)=4.0 mm. With equal handle circumference, doubling lead halves IMA, trading force advantage for greater advance per turn."}},
    37: {"target_skill": "infer an efficiency trend from machine performance data", "novel_context": "block-and-tackle load test",
         "fixed_givens": ["pulley IMA is 4 for every trial", "load-effort pairs are 200 N and 65 N, 400 N and 120 N, 600 N and 175 N",
                           "each load rises at constant speed", "efficiency equals (load/effort)/IMA"],
         "verification": {"expected_answer": "efficiency increases with load: about 76.9%, 83.3%, and 85.7%",
                          "calculation": "Efficiencies are (200/65)/4=76.9%, (400/120)/4=83.3%, and (600/175)/4=85.7%, so the measured efficiency rises across the trials."}},
    38: {"target_skill": "choose a force-distance tradeoff under a design constraint", "novel_context": "manual rescue hoist",
         "fixed_givens": ["current ideal hoist IMA is 3", "operator cannot supply enough force", "there is room to pull more rope", "load and lift height do not change"],
         "verification": {"expected_answer": "add supporting strands to increase IMA, reducing force while requiring more rope travel",
                          "calculation": "A larger pulley IMA reduces ideal effort force. Conservation of work requires effort distance to increase correspondingly."}},
    39: {"target_skill": "combine gear torque ratio, drum torque, and efficiency", "novel_context": "motorized stage-curtain winch",
         "fixed_givens": ["15-tooth motor gear drives a 45-tooth drum gear", "drum radius 0.080 m", "curtain load 500 N",
                           "overall efficiency from motor shaft to drum is 80%", "constant-speed lifting"],
         "verification": {"expected_answer": "16.7 N m", "expected_value": 16.6666667,
                          "expression": "(500*0.080)/((45/15)*0.80)",
                          "calculation": "The drum needs 500(0.080)=40 N m. Ideal gear torque multiplication is 45/15=3; with 80% efficiency, output torque is 2.4 times motor torque. Required motor torque=40/2.4=16.7 N m."}},
    40: {"target_skill": "predict wheel-and-axle IMA from a geometry change", "novel_context": "interchangeable handwheel on a winch",
         "fixed_givens": ["axle radius stays fixed", "wheel radius is doubled", "ideal wheel-and-axle", "IMA equals wheel radius divided by axle radius"],
         "verification": {"expected_answer": "IMA doubles",
                          "calculation": "With axle radius unchanged, IMA=R_wheel/R_axle is directly proportional to wheel radius, so doubling the wheel radius doubles IMA."}},
    41: {"target_skill": "compare wedge IMA from proportions", "novel_context": "two splitting wedges",
         "fixed_givens": ["both wedges have the same thickness", "Wedge B is longer than Wedge A", "ideal wedge IMA equals length divided by thickness"],
         "verification": {"expected_answer": "Wedge B has greater IMA and needs less ideal effort force",
                          "calculation": "At fixed thickness, the longer wedge has the larger length/thickness ratio. Its greater IMA reduces ideal effort for the same output force."}},
    42: {"target_skill": "classify a third-class lever from component order", "novel_context": "student using tweezers",
         "fixed_givens": ["joined end of tweezers is the fulcrum", "fingers apply effort along the arms", "object at the tips is the load", "effort lies between fulcrum and load"],
         "verification": {"expected_answer": "third-class lever",
                          "calculation": "A third-class lever places effort between the fulcrum and the load, matching the tweezers arrangement."}},
    43: {"target_skill": "derive compound IMA and test independent force and travel constraints", "novel_context": "lever-driven Weston differential rescue hoist",
         "fixed_givens": ["ideal lever effort arm 0.90 m and load arm 0.15 m", "lever output pulls a Weston hoist with R=0.18 m and r=0.15 m",
                           "Weston IMA=2R/(R-r)", "overall efficiency 55%", "load is 1000 N and target rise is 0.25 m",
                           "operator effort is limited to 40 N and effort travel to 12 m"],
         "verification": {"expected_answer": "overall IMA 72; 25.3 N required so the force limit passes; 18.0 m travel required so the distance limit fails; maximum rise is 0.167 m",
                          "calculation": "Lever IMA=0.90/0.15=6 and Weston IMA=0.36/0.03=12, so total IMA=72. AMA=0.55(72)=39.6 and effort=1000/39.6=25.3 N. Geometry requires 72(0.25)=18 m, above 12 m. Maximum rise=12/72=0.167 m."}},
    44: {"target_skill": "optimize machine choice under real force and geometric distance constraints", "novel_context": "four candidate loading-hoist designs",
         "fixed_givens": ["900 N load rises 0.40 m", "effort force limit 100 N", "effort travel limit 4.0 m",
                           "A: IMA 8 and 75% efficiency", "B: IMA 12 and 80% efficiency", "C: IMA 10 and 90% efficiency",
                           "D: IMA 9 and 95% efficiency", "effort=load/(IMA times efficiency), effort travel=IMA times load rise"],
         "verification": {"expected_answer": "Design C only",
                          "calculation": "A needs 150 N and 3.2 m; B needs 93.75 N and 4.8 m; C needs 100 N and 4.0 m; D needs 105.3 N and 3.6 m. Only C meets both limits."}},
    45: {"target_skill": "connect angle of repose, friction coefficients, and post-slip acceleration", "novel_context": "adjustable materials-test ramp",
         "fixed_givens": ["block first slips at 35 degrees", "use tan35=0.700, sin35=0.574, cos35=0.819", "kinetic friction coefficient is 0.50",
                           "after slipping the ramp remains at 35 degrees", "use g=9.8 m/s^2"],
         "verification": {"expected_answer": "static coefficient 0.700 and downhill acceleration 1.61 m/s^2",
                          "calculation": "At impending slip, μs=tan35=0.700. Once moving, a=g(sin35-μk cos35)=9.8[0.574-0.50(0.819)]=1.61 m/s^2 downhill."}},
    46: {"target_skill": "recognize both conditions for static equilibrium", "novel_context": "stationary lever-supported shop sign",
         "fixed_givens": ["sign and support are completely at rest", "support may have several forces at different locations", "system is treated as a rigid body"],
         "verification": {"expected_answer": "net force is zero and net torque is zero",
                          "calculation": "A rigid body is in static equilibrium only when translational acceleration and angular acceleration are both zero, requiring ΣF=0 and Στ=0."}},
    47: {"target_skill": "apply the angular dependence of torque qualitatively", "novel_context": "loosening a bolt with a wrench",
         "fixed_givens": ["same wrench length", "same force magnitude", "force may be applied at different angles to the handle", "torque magnitude is r F sin(theta)"],
         "verification": {"expected_answer": "90 degrees to the handle",
                          "calculation": "For fixed r and F, torque is largest when sin(theta)=1, which occurs when the force is perpendicular to the wrench handle."}},
    48: {"target_skill": "solve torque equilibrium and two-axis hinge reactions", "novel_context": "cable-supported horizontal drawbridge",
         "fixed_givens": ["uniform 4.0 m bridge hinged at its left end", "bridge weight 600 N acts 2.0 m from hinge", "400 N cart is 3.0 m from hinge",
                           "cable attaches at right end and makes 30 degrees above bridge", "use sin30=0.500 and cos30=0.866", "static equilibrium"],
         "verification": {"expected_answer": "cable tension 1200 N; hinge force 1040 N right and 400 N up",
                          "calculation": "Torque about hinge: 4T sin30=600(2)+400(3), so T=1200 N. The cable pulls left with T cos30=1039 N and up with 600 N. Force balance therefore requires hinge reactions 1039 N right and 1000-600=400 N up."}},
    49: {"target_skill": "calculate AMA from measured forces", "novel_context": "tested crate-lifting machine",
         "fixed_givens": ["machine lifts a 480 N load", "measured effort is 120 N", "constant-speed motion", "AMA equals load force divided by effort force"],
         "verification": {"expected_answer": "4", "expected_value": 4,
                          "expression": "480/120", "calculation": "AMA=load/effort=480/120=4."}},
    50: {"target_skill": "calculate efficiency from input and output work", "novel_context": "work measurements on a shop hoist",
         "fixed_givens": ["input work 500 J", "useful output work 350 J", "efficiency equals output work divided by input work times 100%"],
         "verification": {"expected_answer": "70%", "expected_value": 70,
                          "expression": "350/500*100", "calculation": "Efficiency=(350 J/500 J)(100%)=70%."}},
    51: {"target_skill": "multiply stage IMAs for an ideal compound machine", "novel_context": "lever driving a block-and-tackle",
         "fixed_givens": ["ideal lever IMA is 3", "lever drives an ideal pulley system with IMA 4", "stages act in series"],
         "verification": {"expected_answer": "12", "expected_value": 12,
                          "expression": "3*4", "calculation": "For ideal stages in series, total IMA is the product: 3(4)=12."}},
    52: {"target_skill": "compare screw lead angle with friction angle to predict back-driving", "novel_context": "lubricated screw jack safety check",
         "fixed_givens": ["screw mean radius 10 mm", "single-start lead 2.0 mm", "tan(lead angle)=lead/(2 pi radius)",
                           "use pi=3.14", "dry friction coefficient 0.080", "lubricated friction coefficient 0.020",
                           "use the rule that the screw is self-locking when friction coefficient exceeds tan(lead angle)"],
         "verification": {"expected_answer": "tan(lead angle)=0.0318; dry screw is self-locking, lubricated screw can back-drive",
                          "calculation": "tan λ=2.0/[2(3.14)(10)]=0.0318. Dry μ=0.080 exceeds 0.0318, so it self-locks. Lubricated μ=0.020 is smaller, so the load can back-drive the screw."}},
    53: {"target_skill": "calculate useful output power", "novel_context": "small platform lift",
         "fixed_givens": ["lift raises a 300 N load", "vertical distance 2.0 m", "elapsed time 15 s", "power equals useful work divided by time"],
         "verification": {"expected_answer": "40 W", "expected_value": 40,
                          "expression": "300*2.0/15", "calculation": "Useful work=300(2.0)=600 J, so output power=600/15=40 W."}},
    54: {"target_skill": "apply differential-hoist IMA and efficiency", "novel_context": "Weston hoist load test",
         "fixed_givens": ["joined upper sheave radii R=20 cm and r=16 cm", "Weston IMA=2R/(R-r)",
                           "overall efficiency 70%", "load 840 N", "constant-speed lift"],
         "verification": {"expected_answer": "IMA 10 and effort 120 N",
                          "calculation": "IMA=2(20)/(20-16)=10. AMA=0.70(10)=7, so effort=840/7=120 N."}},
    55: {"target_skill": "distinguish geometric IMA from friction-dependent AMA and efficiency", "novel_context": "lever before and after fulcrum corrosion",
         "fixed_givens": ["lever arm distances do not change", "fulcrum friction increases", "same load is lifted slowly", "IMA depends only on geometry"],
         "verification": {"expected_answer": "IMA stays the same, while AMA and efficiency decrease",
                          "calculation": "Unchanged arm lengths preserve geometric IMA. Extra friction requires greater effort for the same load, reducing load/effort (AMA) and therefore AMA/IMA efficiency."}},
    56: {"target_skill": "apply wheel-and-axle force-distance tradeoff", "novel_context": "winch with a replacement rope drum",
         "fixed_givens": ["wheel radius stays fixed", "axle radius is doubled", "ideal wheel-and-axle", "same load and lift height"],
         "verification": {"expected_answer": "IMA is halved, required effort doubles, and each turn lifts the load twice as far",
                          "calculation": "IMA=Rwheel/Raxle, so doubling axle radius halves IMA and doubles ideal effort for a fixed load. Axle circumference doubles, so one turn winds twice as much rope."}},
    57: {"target_skill": "compare torque products for perpendicular forces", "novel_context": "two wrench setups on the same bolt",
         "fixed_givens": ["original force 80 N perpendicular at 0.30 m from bolt", "comparison forces are perpendicular", "torque equals force times lever arm"],
         "verification": {"expected_answer": "60 N at 0.40 m",
                          "calculation": "Original torque=80(0.30)=24 N m. The matching setup is 60(0.40)=24 N m."}},
    58: {"target_skill": "apply screw pitch force-distance tradeoff", "novel_context": "two otherwise identical bench-vise screws",
         "fixed_givens": ["same handle radius", "Screw A pitch 2 mm", "Screw B pitch 4 mm", "both single-start and ideal"],
         "verification": {"expected_answer": "B advances twice as far per turn but has half A's IMA, so it needs twice the ideal effort for the same load",
                          "calculation": "Lead equals pitch here, so B advances 4/2=2 times as far. Screw IMA is handle circumference/lead, so doubling lead halves IMA and doubles ideal effort."}},
    59: {"target_skill": "distinguish absolute work loss from percent efficiency", "novel_context": "two block-and-tackle performance trials",
         "fixed_givens": ["both pulley systems have IMA 4 and lift loads 1.0 m", "effort rope travels 4.0 m",
                           "System A lifts 400 N with 120 N effort", "System B lifts 600 N with 170 N effort"],
         "verification": {"expected_answer": "Both lose 80 J, but System B is more efficient",
                          "calculation": "A input=120(4)=480 J and output=400 J, so loss=80 J and efficiency=83.3%. B input=170(4)=680 J and output=600 J, so loss=80 J and efficiency=88.2%."}},
    60: {"target_skill": "distinguish work from power for the same lift", "novel_context": "two runs of an ideal winch",
         "fixed_givens": ["same load", "same vertical height", "Run B takes half the time of Run A", "both winch runs are ideal"],
         "verification": {"expected_answer": "Both do the same useful work, but Run B has twice the output power",
                          "calculation": "Same load and height give the same work. Power is work/time, so completing that work in half the time doubles power."}},
}


def vetted_plan(index: int, topic: str, difficulty: int, response_type: str, brief: str) -> dict[str, Any]:
    plan = json.loads(json.dumps(VETTED_PLANS[index]))
    givens = plan["fixed_givens"]
    nodes = [{"id": f"g{number}", "type": "Given", "label": value} for number, value in enumerate(givens, 1)]
    nodes += [{"id": "law", "type": "Law", "label": plan["target_skill"]},
              {"id": "target", "type": "Target", "label": plan["verification"]["expected_answer"]}]
    plan.update({"response_type": response_type, "topic": topic, "difficulty": difficulty,
                 "required_brief": brief, "reasoning_graph": {"nodes": nodes,
                 "edges": ([{"src": node["id"], "dst": "law", "type": "supports"} for node in nodes[:-2]] +
                           [{"src": "law", "dst": "target", "type": "derived_from"}])},
                 "distractor_mechanisms": ["invert a ratio", "ignore one constraint or stage", "treat a real machine as ideal"],
                 "audit": {"valid": True, "method": "vetted_specification"}})
    return plan


def choose_anchor(corpus: list[dict[str, Any]], topic: str, response_type: str, offset: int) -> dict[str, Any]:
    candidates = [row for row in corpus if row.get("response_type") == response_type and topic in (row.get("topics") or [])]
    if not candidates:
        candidates = [row for row in corpus if row.get("response_type") == response_type]
    if not candidates:
        raise RuntimeError(f"No {response_type} source anchor for {topic}")
    return candidates[offset % len(candidates)]


def main() -> None:
    parser = argparse.ArgumentParser(description="Replace weak Machines items with targeted challenge items")
    parser.add_argument("--run-dir", default="science_olympiad/machines_b/final")
    parser.add_argument("--model", default="gpt-5.1")
    parser.add_argument("--embedding-model", default="text-embedding-3-small")
    parser.add_argument("--provider", choices=["openai", "ollama"], default="openai")
    parser.add_argument("--seed", type=int, default=11900)
    args = parser.parse_args()
    root = Path(args.run_dir)
    by_id = {row["id"]: row for row in read_jsonl(root / "corpus" / "items.jsonl")}
    q_rows = read_jsonl(root / "enriched" / "question_embeddings.jsonl")
    s_rows = read_jsonl(root / "enriched" / "structure_embeddings.jsonl")
    corpus = [by_id[row["id"]] for row in q_rows]
    source_qmat = normalized([row["embedding"] for row in q_rows])

    original = read_jsonl(root / "generated" / "items.jsonl")
    base_items = [row for row in original if row["id"] not in REJECT_IDS]
    original_reports = {row["id"]: row for row in read_jsonl(root / "validation" / "item_reports.jsonl")}
    items = list(base_items)
    reports = [original_reports[row["id"]] for row in items]
    plans = [(row.get("generation") or {}).get("plan") or {} for row in items]

    out_items = root / "generated" / "items_curated.jsonl"
    out_plans = root / "generated" / "reasoning_plans_curated.jsonl"
    out_reports = root / "validation" / "item_reports_curated.jsonl"
    if out_items.exists():
        saved = read_jsonl(out_items)
        if len(saved) >= len(base_items):
            saved_reports, saved_plans = read_jsonl(out_reports), read_jsonl(out_plans)
            kept = [(item, report, plan) for item, report, plan in zip(saved, saved_reports, saved_plans)
                    if item["id"] not in REJECT_IDS | REGENERATE_IDS]
            items = [row[0] for row in kept]
            reports = [row[1] for row in kept]
            plans = [row[2] for row in kept]
    prior_vectors = embed_texts(args.embedding_model, [row["prompt"] for row in items], provider=args.provider)

    existing_ids = {item["id"] for item in items}
    for index in range(len(CHALLENGES)):
        serial = 101 + index
        generated_id = f"machines-b-generated-{serial:03d}"
        if generated_id in REJECT_IDS or generated_id in existing_ids:
            continue
        topic, difficulty, response_type, brief = CHALLENGES[index]
        anchor = choose_anchor(corpus, topic, response_type, index)
        exemplars = retrieve_exemplars(anchor, corpus, q_rows, s_rows)
        plan = (vetted_plan(index, topic, difficulty, response_type, brief) if index in VETTED_PLANS else
                make_plan(anchor, exemplars, topic, difficulty, plans, args.model, args.provider,
                          args.seed + index * 1000, required_brief=brief))
        item, report, vector = create_item(anchor, exemplars, plan, items, source_qmat, corpus, prior_vectors,
                                           args.embedding_model, args.model, args.provider,
                                           args.seed + index * 1000 + 300, serial)
        plans.append(plan)
        items.append(item)
        reports.append(report)
        prior_vectors.append(vector)
        existing_ids.add(item["id"])
        write_jsonl(out_plans, plans)
        write_jsonl(out_items, items)
        write_jsonl(out_reports, reports)
        print(f"Accepted targeted replacement {index + 1}/{len(CHALLENGES)}: {topic}", flush=True)

    payload = [{"id": row["id"], "response_type": row["response_type"], "difficulty": row["difficulty"],
                "topics": row["topics"], "prompt": row["prompt"], "choices": row.get("choices")} for row in items]
    calibration = [{"response_type": row.get("response_type"), "difficulty": row.get("difficulty"),
                    "prompt": row.get("prompt"), "choices": row.get("choices")} for row in corpus[::max(1, len(corpus)//16)][:16]]
    audit_prompt = "REAL 2026 WRITTEN-TEST CALIBRATION EXAMPLES:\n" + json.dumps(calibration, ensure_ascii=False) + \
                   "\nGENERATED QUESTION SET:\n" + json.dumps(payload, ensure_ascii=False)
    audit = generate_json(args.model, audit_prompt, provider=args.provider,
                          system=BATCH_AUDIT_SYSTEM, temperature=0, max_output_tokens=5000, seed=args.seed + 999999)
    (root / "validation" / "batch_audit_curated.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Curated {len(items)} items; batch audit overall_good={audit.get('overall_good')}")


if __name__ == "__main__":
    main()
