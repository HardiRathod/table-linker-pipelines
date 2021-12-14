

1. Download Data From NIH Reporter and Convert to KGTK files

	Input: 
	nih_raw_file
	https://drive.google.com/drive/u/0/folders/12dW8hOTKS-iaJINc6KjWrs7gfyf0h4I6

	from
	https://exporter.nih.gov/ExPORTER_Catalog.aspx

	Run 1_NIH_Reporter_to_KGTK_files.ipynb notebook (in processing folder) to convert data from NIH Reporter csv files to KGTK files

	https://github.com/usc-isi-i2/table-linker-pipelines/blob/main/nih/processing/1_NIH_Reporter_to_KGTK_files.ipynb

	Results: nih_kgtk_files 
	https://drive.google.com/drive/u/0/folders/1UKwiTVef5yqhsUD454h_inYqjcl5Pcqu



2.  Converting Raw KGTK Files to Table Linker

	Input: nih_kgtk_files 
	https://drive.google.com/drive/u/0/folders/1UKwiTVef5yqhsUD454h_inYqjcl5Pcqu

	Run 2_KGTK_files_to_tl_and_tableau.ipynb notebook (in processing folder) to convert raw files to KGTK files

	https://github.com/usc-isi-i2/table-linker-pipelines/blob/main/nih/processing/2_KGTK_files_to_tl_and_tableau.ipynb

	Final outputs are:

	Files for table-linker 
	https://drive.google.com/drive/u/0/folders/1eKHhUl2R6-Rbh5tf0eRecNLkkthvV-I4


3. Run table-linker for organization and people:

	Input: Table_Linker_Complete_Files: 
	https://drive.google.com/drive/u/0/folders/1eKHhUl2R6-Rbh5tf0eRecNLkkthvV-I4

	Run: Table_linker_pipelines
	https://github.com/usc-isi-i2/table-linker-pipelines/blob/main/nih/org-pipeline.ipynb
	https://github.com/usc-isi-i2/table-linker-pipelines/blob/main/nih/person-pipeline.ipynb

	Results: person_with_qid 
	https://drive.google.com/drive/u/1/folders/10QsHjFFdJp2yMWjldDXqC7zy8uDbxc7z

	organization_with_qid 
	https://drive.google.com/drive/u/1/folders/1W1ll4h34fhDHyd9GDP7y1Qc-GSR7y-NY



4. Converting Raw KGTK Files Tableau Files

	Input: 
	Wikidata Files (steps 1-4) in Build ElasticSearch Index Files for Table Linker, 
	https://docs.google.com/document/d/15CDuNgixRpDQmeEo7XImPHBtla72hvJ0fniS_BGu9Bk/edit#heading=h.m95wofmy28v

	nih_kgtk_files
	https://drive.google.com/drive/u/0/folders/1UKwiTVef5yqhsUD454h_inYqjcl5Pcqu

	nih_project_files https://drive.google.com/drive/u/0/folders/1yeo7e0KwqDPddzdAr43zutZYsmu8T3c4

	person_with_qid https://drive.google.com/drive/u/1/folders/10QsHjFFdJp2yMWjldDXqC7zy8uDbxc7z

	organization_with_qid https://drive.google.com/drive/u/1/folders/1W1ll4h34fhDHyd9GDP7y1Qc-GSR7y-NY

	cluster_coor.tsv
	https://github.com/usc-isi-i2/table-linker-pipelines/blob/main/nih/cluster_coor.tsv

	Run 3_Output_tableau.ipynb notebook to convert the input files for tableau input tsv file

	https://github.com/usc-isi-i2/table-linker-pipelines/blob/main/nih/processing/3_Output_tableau_Files.ipynb

	Final outputs are:

	An excel file for tableau:  nih_0909_new.tsv (in nih folder, created through kgtk community-detection command)

	https://github.com/usc-isi-i2/table-linker-pipelines/blob/main/nih/nih_0909_new.tsv 

5. Visualizating Output File in Tableau 
	For instruction in map visualization with background color for county, 
	see Video: https://youtu.be/XZGLJhLFZQ0
