import streamlit as st
import pandas as pd
import numpy as np
import rdkit
from rdkit import Chem
from rdkit.Chem import QED

def QEDcalculation(query):
    """
    간단한 QED 계산기. 
    query로 SMILES를 받아 계산이나 해보자.
    """
    m = Chem.MolFromSmiles(query)
    result = "QED" + str(round(QED.qed(m),3)) + "WLOGP" + str(round(Chem.Crippen.MolLogP(m),2))
    
    return result
    

input_string  = st.text_input("Please input interesting SMILES","CC(=C)C(O)=O")
st.write("현재", input_string)
st.write(QEDcalculation(input_string))

if input_string == True:
    st.write(QEDcalculation(input_string)) #메타크릴산 독성.
