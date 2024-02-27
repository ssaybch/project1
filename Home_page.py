import streamlit as st
import pandas as pd
import numpy as np
import rdkit
from rdkit import Chem
from rdkit.Chem import QED, Descriptors, Lipinski

# Chembl 예측 관련 모듈
import onnxruntime
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
from chembl_webresource_client.new_client import new_client

# mol 그리기
from streamlit_ketcher import st_ketcher



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
ort_session = onnxruntime.InferenceSession("./chembl_33_multitask.onnx")

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
    for m in range(len(tars)):
        if tars[m]['organism'] == 'Homo sapiens':
            for n in tars[m]['target_components'][0]['target_component_synonyms']:
                if n['syn_type'] == "GENE_SYMBOL":
                    gene_list = n['component_synonym']
                    result_9606_GENE_chemblid_proba.append([
                        gene_list,
                        tars[m]['target_chembl_id'],
                        dict_predicted_CHEMBL[tars[m]['target_chembl_id']]
                    ])
    result_df = pd.DataFrame(data = result_9606_GENE_chemblid_proba, columns=['GENE','ChEMBL_ID','probability'])
    return result_df
            

##### 사이드바 지정 단락 #####
with st.sidebar:
    st.write("여기는 사이드바")


##### 메인콘텐츠 지정 단락 #####
input_string  = st.text_input("Please input interesting SMILES","CC(=C)C(O)=O", help="올바르지 않은 SMILES일 경우 에러가 출력됩니다.")
st.write("입력한 분자 SMILES: ", input_string)
smile_code = st_ketcher(input_string, height=400)

mw, qed, wlogp, tpsa, hbd, hba, rtb, violation, charge = calc_rdkit(input_string)
st.write("Molecular weight: ", mw)
st.write("QED: ", qed)
st.write("wlogp: ", wlogp)
st.write("TPSA; ", tpsa, "(단, S와 P의 영향은 고려하지 않았으며 SwissADME의 결과와 동일함.)")
st.write("num of Hydrogen bond donors: ", hbd)
st.write("num of Hydrogen bond acceptors: ", hba)
st.write("num of Rotatable bonds: ", rtb)
st.write("Ro5: ", violation)
st.write("Charge: ", charge)

st.dataframe(chembl_func(input_string))

