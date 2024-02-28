import streamlit as st
import pandas as pd
import numpy as np
import requests
import rdkit
from rdkit import Chem
from rdkit.Chem import QED, Descriptors, Lipinski
import json

# Chembl 예측 관련 모듈
import onnxruntime
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
from chembl_webresource_client.new_client import new_client

# mol 그리기
from streamlit_ketcher import st_ketcher

# PDB 그리기
from stmol import *
import py3Dmol


##### 페이지레이아웃 지정 단락 #####
st.set_page_config(
    page_title="PT2 WEB-app",
    layout="wide"
)

##### 함수 지정 단락 #####
def ro5(mw, hbd, hba, wlogp):
    ro5_violations = 0
    failed = list()
    if mw >= 500:
        ro5_violations = ro5_violations + 1
        failed.append("molecular weight violated: %s" % mw)
    if hbd > 5:
        ro5_violations = ro5_violations + 1
        failed.append("HBD violated: %s" % hbd)
    if hba > 10:
        ro5_violations = ro5_violations + 1
        failed.append("HBA violated: %s" % hba)
    if wlogp > 5:
        ro5_violations = ro5_violations + 1 #초과냐 이상이냐

    if ro5_violations > 0:
        violation = "violated"
    else:
        violation = "위반사항이 없습니다."
    
    return violation
    

def calc_rdkit(query):
    """
    간단한 SMILES 계산기.
    rdkit으로 물리화학적 특성을 반환시킴
    mw    : ExactMolWt과 약간 다르다.
    qed   : https://doi.org/10.1038/nchem.1243
    wlogp : S.A. Wildman and G. M. Crippen JCICS 39 868-873 (1999)
    tpsa  : https://doi.org/10.1021/jm000942e
            Swiss ADME에서 계산하는 TPSA는 Etrl 2000. 에 기반한다. RDKit은 이 참고문헌에 S와 P의 영향을 함께 계산할 수 있는 옵션을 제공한다. (includeSandP=True) 
            includeSandP 옵션을 켜지 않으면 SwissADME에서 계산한 결과와 동일하다.
    Ro5   : https://doi.org/10.1016/S0169-409X(96)00423-1
    lipinski module : https://www.rdkit.org/docs/source/rdkit.Chem.Lipinski.html?highlight=lipinski%20module
    """

    mol = Chem.MolFromSmiles(query)
    mw = Descriptors.MolWt(mol)
    qed = round(QED.qed(mol),3)
    wlogp = Descriptors.MolLogP(mol) #Chem.Crippen.MolLogP(mol)과 같은 기능
    tpsa = Descriptors.TPSA(mol, includeSandP = False)
    hbd = Lipinski.NumHDonors(mol)
    hba = Lipinski.NumHAcceptors(mol)
    rtb = Lipinski.NumRotatableBonds(mol)
    violation = ro5(mw, hbd, hba, wlogp)
    charge = Chem.rdmolops.GetFormalCharge(mol) 

    return mw, qed, wlogp, tpsa, hbd, hba, rtb, violation, charge


def calc_morgan_fp(smiles):
    mol = Chem.MolFromSmiles(smiles)
    fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(
        mol, 2, nBits=1024)
    a = np.zeros((0,), dtype=np.float32)
    Chem.DataStructs.ConvertToNumpyArray(fp, a)
    return a

def format_preds(preds, targets):
    preds = np.concatenate(preds).ravel()
    np_preds = [(tar, pre) for tar, pre in zip(targets, preds)]
    dt = [('chembl_id','|U20'), ('pred', '<f4')]
    np_preds = np.array(np_preds, dtype=dt)
    np_preds[::-1].sort(order='pred')
    return np_preds



# CHEMBL예측 미리 불러올것 밖으로 빼놓기
ort_session = onnxruntime.InferenceSession("chembl_33_multitask.onnx")

def chembl_func(smiles):
    descs = calc_morgan_fp(smiles)
    ort_inputs = {ort_session.get_inputs()[0].name: descs}
    preds = ort_session.run(None, ort_inputs)
    
    preds = format_preds(preds, [o.name for o in ort_session.get_outputs()])
    filtered_preds = np.array([item for item in preds if float(item[1]) >= 0.7])
    
    predicted_CHEMBL = [str(n[0]) for n in filtered_preds] #예측값하한선 0.7
    predicted_probability = [float(n[1]) for n in filtered_preds] #예측값하한선 0.7
    dict_predicted_CHEMBL = dict(zip(predicted_CHEMBL, predicted_probability))
    
    target = new_client.target
    tars = target.filter(target_chembl_id__in=list(dict_predicted_CHEMBL.keys()))
    
    result_9606_GENE_chemblid_proba = list()
    result_uniprot = list()
    for m in range(len(tars)):
        if tars[m]['organism'] == 'Homo sapiens':
            for n in tars[m]['target_components'][0]['target_component_synonyms']:
                if n['syn_type'] == "GENE_SYMBOL":
                    gene_list = n['component_synonym']
                    result_9606_GENE_chemblid_proba.append([
                        gene_list,
                        tars[m]['target_chembl_id'],
                        "https://www.uniprot.org/uniprotkb/" + tars[m]['target_components'][0]['accession'] + "/entry",
                        dict_predicted_CHEMBL[tars[m]['target_chembl_id']]
                    ])
                    result_uniprot.append(tars[m]['target_components'][0]['accession'])
    result_df = pd.DataFrame(data = result_9606_GENE_chemblid_proba, columns=['GENE','ChEMBL_ID','UNIPROT_ID','probability'])
    return result_df, result_uniprot


def is_point_inside_ellipse(x, y, ellipse_center, ellipse_width, ellipse_height, angle):
    # 타원의 중심, 너비, 높이, 회전 각도를 변수로 받음
    h, k = ellipse_center
    a, b = ellipse_width / 2, ellipse_height / 2
    theta = np.radians(angle)  # 각도를 라디안으로 변환

    # 점 (x, y)를 타원 중심을 기준으로 회전
    x_rot = (x - h) * np.cos(-theta) + (y - k) * np.sin(-theta)
    y_rot = -(x - h) * np.sin(-theta) + (y - k) * np.cos(-theta)

    # 변환된 좌표를 사용하여 타원 내부에 있는지 확인
    if (x_rot**2 / a**2) + (y_rot**2 / b**2) < 1:
        return True  # 타원 내부에 있음
    else:
        return False  # 타원 외부에 있음

ellipse_center_white_yolk = (71.051, 2.292)
ellipse_width_white_yolk = 142.081
ellipse_height_white_yolk = 8.740
angle_white_yolk = -1.031325

ellipse_center_yolk = (38.117, 3.177)
ellipse_width_yolk = 82.061
ellipse_height_yolk = 5.557
angle_yolk = -0.177887

# 점이 yolk 타원 내부에 있는지 확인
def is_inside_EGG(x, y):
    HIA = is_point_inside_ellipse(
        x, y, 
        ellipse_center_white_yolk, ellipse_width_white_yolk, ellipse_height_white_yolk, 
        angle_white_yolk
    )
    
    BBB = is_point_inside_ellipse(
        x, y, 
        ellipse_center_yolk, ellipse_width_yolk, ellipse_height_yolk, 
        angle_yolk
    )
    return HIA, BBB


# PDB_id 받아오기
def uniprot_to_pdb(query):
    url = "https://rest.uniprot.org/uniprotkb/" + query
    req = requests.get(url)
    soup = json.loads(req.text)
    pdb_list = [item['id'] for item in soup['uniProtKBCrossReferences'] if item['database'] == 'PDB']
    return pdb_list


# PDB 렌더
def render_pdb(id='7T59'):
    viewer = py3Dmol.view(query=id)
    viewer.setStyle({ "cartoon": {
        "color": "spectrum",
        "colorReverse": True,
        "colorScale": "RdYlGn",
        "colorScheme": "Polarity",
        "colorBy": "resname",
            }})
    return viewer


##### 사이드바 지정 단락 #####
with st.sidebar:
    st.write("asdfasd")


##### 메인콘텐츠 지정 단락 #####
st.subheader("분석하려는 성분의 SMILES 또는 isoSMILES를 아래에 입력하세요.")
input_string  = st.text_input("","CC(=C)C(O)=O", help="올바르지 않은 SMILES일 경우 에러가 출력됩니다.")
st.write("입력한 분자 SMILES: ", input_string)
smile_code = st_ketcher(input_string, height=400)

mw, qed, wlogp, tpsa, hbd, hba, rtb, violation, charge = calc_rdkit(input_string)
HIA, BBB = is_inside_EGG(tpsa, wlogp)
if HIA == True:
    HIA = 'High'
else:
    HIA = 'Low'

result_df, result_uniprot = chembl_func(input_string)

st.subheader("Physicochemical properties, simple ADME")
st.write("Molecular weight: ", round(mw, 3))
st.write("QED: ", qed, '[1]')
st.write("wlogp: ", round(wlogp, 3), '[2]')
st.write("TPSA; ", tpsa, "(단, S와 P의 영향은 고려하지 않았으며 SwissADME의 결과와 동일함.)")
st.write("num of Hydrogen bond donors: ", hbd)
st.write("num of Hydrogen bond acceptors: ", hba)
st.write("num of Rotatable bonds: ", rtb)
st.write("Charge: ", charge)
st.write("Ro5: ", violation, '[3]')
st.write("Human intestine absorbable: ", HIA, '[4]')
st.write("Brain-blood barrier permeable: ", BBB, '[4]')

st.subheader("Target protein prediction")
st.text("관심 성분과 70 % 이상의 확률로 결합이 예측되는 단백질은 다음과 같습니다. ChEMBL DB 33 버전을 사용합니다.")
st.dataframe(result_df,
             use_container_width =True,
             column_config={
                 "UNIPROT_ID": st.column_config.LinkColumn(
                     display_text="https://www.uniprot.org/uniprotkb/(.*?)/entry"
                 )
             }
            )

option_uniprot = st.selectbox(
    'UNIPROT ID골라요',
    (result_uniprot),
    index=None,
    placeholder='UNIPROT ID골라요',
)
st.write(option_uniprot)

if option_uniprot != 'UNIPROT ID골라요':
    pdb_list = uniprot_to_pdb(option_uniprot)

option_pdb = st.selectbox(
    'PDB ID골라요',
    (pdb_list),
    index=None,
    placeholder='PDB 골라요',
)

if option_pdb != 'PDB 골라요':
    showmol(render_pdb(id = option_pdb))

# xyzview = py3Dmol.view(query='pdb:1A2C') 
# xyzview.setStyle({'cartoon':{'color':'spectrum'}})
# showmol(xyzview, height = 500,width=800)



st.write("")
st.write("References")
st.write("[1] Bickerton GR, Paolini GV, Besnard J, Muresan S, Hopkins AL. Quantifying the chemical beauty of drugs. Nat Chem. 2012 Jan 24;4(2):90-8. doi: 10.1038/nchem.1243. PMID: 22270643; PMCID: PMC3524573.")
st.write("[2] Wildman SA, Crippen GM. Prediction of Physicochemical Parameters by Atomic Contributions. J. Chem. Inf. Comput. Sci. 1999 Aug 19;39(5):868-73. doi: 10.1021/ci990307l.")
st.write("[3] Lipinski CA, Lombardo F, Dominy BW, Feeney PJ. Experimental and computational approaches to estimate solubility and permeability in drug discovery and development settings. Adv Drug Deliv Rev. 2001 Mar 1;46(1-3):3-26. doi: 10.1016/s0169-409x(00)00129-0. PMID: 11259830.")
st.write("[4] Daina A, Zoete V. A BOILED-Egg To Predict Gastrointestinal Absorption and Brain Penetration of Small Molecules. ChemMedChem. 2016 Jun 6;11(11):1117-21. doi: 10.1002/cmdc.201600182. Epub 2016 May 24. PMID: 27218427; PMCID: PMC5089604.")
