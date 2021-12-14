

1. Download Data From NIH Reporter and Convert to KGTK files

	Input:  
	
	https://exporter.nih.gov/ExPORTER_Catalog.aspx
	
	Run 1_NIH_Reporter_to_KGTK_files.ipynb notebook to convert data from NIH Reporter csv files to KGTK files
	Results: nih_kgtk_files 
	https://drive.google.com/drive/u/0/folders/1UKwiTVef5yqhsUD454h_inYqjcl5Pcqu

	
	
2.  Converting Raw KGTK Files to Table Linker / Tableau Files

	Input: nih_kgtk_files 
	https://drive.google.com/drive/u/0/folders/1UKwiTVef5yqhsUD454h_inYqjcl5Pcqu
	
	Run 2_KGTK_files_to_tl_and_tableau.ipynb notebook
	
	Final outputs are:
	
	Files for table-linker 
	https://drive.google.com/drive/u/0/folders/1eKHhUl2R6-Rbh5tf0eRecNLkkthvV-I4


3. Run table-linker for organization and people:

	Input: Table_Linker_Complete_Files: 
	https://drive.google.com/drive/u/0/folders/1eKHhUl2R6-Rbh5tf0eRecNLkkthvV-I4
	
	Run: Table_linker_pipelines
	
	Results: person_with_qid 
	https://drive.google.com/drive/u/1/folders/10QsHjFFdJp2yMWjldDXqC7zy8uDbxc7z
	
	organization_with_qid 
	https://drive.google.com/drive/u/1/folders/1W1ll4h34fhDHyd9GDP7y1Qc-GSR7y-NY



4. Converting Raw KGTK Files to Table Linker / Tableau Files

	Input: 
	Wikidata Files (steps 1-4) in Build ElasticSearch Index Files for Table Linker, 
		
	nih_kgtk_files
		
	nih_project_files https://drive.google.com/drive/u/0/folders/1yeo7e0KwqDPddzdAr43zutZYsmu8T3c4
		
	person_with_qid https://drive.google.com/drive/u/1/folders/10QsHjFFdJp2yMWjldDXqC7zy8uDbxc7z
		
	organization_with_qid https://drive.google.com/drive/u/1/folders/1W1ll4h34fhDHyd9GDP7y1Qc-GSR7y-NY
		
	cluster_coor.tsv (in processing folder)
		
	Run 3 tableau.ipynb notebook
	
	Final outputs are:
	
	An excel file for tableau:  nih_0909_new3.xlsx (in nih folder)



Visualizating Output File in Tableau 
	See Video: https://youtu.be/XZGLJhLFZQ0
