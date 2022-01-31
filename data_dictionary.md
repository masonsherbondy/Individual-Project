## Data Dictionary
| #  |  Column                     | Non-Null Count|  Dtype |  Description                                                                                     |
|:---|:----------------------------|:--------------|:-------|:-------------------------------------------------------------------------------------------------|
| 0  |  age                        | 1470 non-null |  int64 | Age of employee (continuous variable)                                                            |
| 1  |  attrition                  | 1470 non-null |  int64 | Whether or not an employee leaves the company. 1: Yes, 0: No                                     | 
| 2  |  daily_rate                 | 1470 non-null |  int64 | Salary level (continuous variable)                                                               |
| 3  |  distance_from_home         | 1470 non-null |  int64 | The distance from work to home (continuous variable)                                             |
| 4  |  education                  | 1470 non-null |  int64 | Education level. 1: Below College, 2: 'College', 3: 'Bachelor's , 4: 'Master , 5: 'Doctorate'    |
| 5  |  environment_satisfaction   | 1470 non-null |  int64 | Level of satisfaction with the environment. 1: 'Low', 2: 'Medium', 3: 'High' , 4: 'Very High'    |
| 6  |  hourly_rate                | 1470 non-null |  int64 | Hourly Salary (continuous variable)                                                              |
| 7  |  job_involvement            | 1470 non-null |  int64 | Numerical value - Job Involvement. 1: 'Low', 2: 'Medium', 3: 'High' , 4: 'Very High'             |
| 8  |  job_level                  | 1470 non-null |  int64 | Level of job (continuous variable)                                                               |
| 9  |  job_satisfaction           | 1470 non-null |  int64 | Level of job satisfaction. 1: 'Low', 2: 'Medium', 3: 'High' , 4: 'Very High'                     |
| 10 |  monthly_income             | 1470 non-null |  int64 | Monthly Salary (continuous variable)                                                             |
| 11 |  monthly_rate               | 1470 non-null |  int64 | Monthly rate (continuous variable)                                                               |
| 12 |  companies_worked           | 1470 non-null |  int64 | Number of companies worked at.                                                                   |
| 13 |  overtime                   | 1470 non-null |  int64 | Overtime Status. 1: Yes, 0: No                                                                   |
| 14 |  percent_salary_hike        | 1470 non-null |  int64 | The percentage of change in salary between 2 year (2017, 2018) (continuous variable)             |
| 15 |  performance_rating         | 1470 non-null |  int64 | Reported Performance Rating. 1: 'Low', 2: 'Good', 3: 'Excellent' , 4: 'Outstanding'              |
| 16 |  relationship_satisfaction  | 1470 non-null |  int64 | Reported level of relationship satisfactioin 1: 'Low', 2: 'Medium', 3: 'High' , 4: 'Very High'   |
| 17 |  stock_option_level         | 1470 non-null |  int64 | Level of stocks owned from company (continuous variable)                                         |
| 18 |  total_working_years        | 1470 non-null |  int64 | Total years worked (continuous variable)                                                         |
| 19 |  hours_trained_last_year    | 1470 non-null |  int64 | Hours spent training (continuous variable)                                                       |
| 20 |  work_life_balance          | 1470 non-null |  int64 | Reported time spent between work and outside. 1: 'Bad', 2: 'Good', 3: 'Better' , 4: 'Best'       |
| 21 |  company_years              | 1470 non-null |  int64 | Total number of years with the company (continuous variable)                                     |
| 22 |  current_role_years         | 1470 non-null |  int64 | Years in current role (continuous variable)                                                      |
| 23 |  years_since_last_promotion | 1470 non-null |  int64 | Years since last promotion (continuous variable)                                                 |
| 24 |  years_with_manager         | 1470 non-null |  int64 | Years spent with current manager (continuous variable)                                           |
| 25 |  travel_none                | 1470 non-null |  uint8 | Dummy variable that represents 0 amount of travel for the company. 1: Yes, 0: No                 |
| 26 |  travel_frequently          | 1470 non-null |  uint8 | Dummy variable that represents frequent amount of travel for the company. 1: Yes, 0: No          |
| 27 |  travel_rarely              | 1470 non-null |  uint8 | Dummy variable that represents rare amount of travel for the company. 1: Yes, 0: No              |
| 28 |  hr_dept                    | 1470 non-null |  uint8 | Dummy variable that represents employee's status in the human resources dept. 1: Yes, 0: No      |
| 29 |  research_dev_dept          | 1470 non-null |  uint8 | Dummy variable that represents employee's status in the research and dev dept. 1: Yes, 0: No     |
| 30 |  sales_dept                 | 1470 non-null |  uint8 | Dummy variable that represents employee's status in the sales department. 1: Yes, 0: No          |
| 31 |  hr_ed                      | 1470 non-null |  uint8 | Dummy variable that represents education status in human resources. 1: Yes, 0: No                |
| 32 |  life_sciences_ed           | 1470 non-null |  uint8 | Dummy variable that represents education status in life sciences. 1: Yes, 0: No                  |
| 33 |  marketing_ed               | 1470 non-null |  uint8 | Dummy variable that represents education status in marketing. 1: Yes, 0: No                      |
| 34 |  medical_ed                 | 1470 non-null |  uint8 | Dummy variable that represents education status in the medical field. 1: Yes, 0: No              |
| 35 |  other_ed                   | 1470 non-null |  uint8 | Dummy variable that represents education status listed as 'other.' 1: Yes, 0: No                 |
| 36 |  tech_deg_ed                | 1470 non-null |  uint8 | Dummy variable that represents education status as having a technical degree. 1: Yes, 0: No      |
| 37 |  female                     | 1470 non-null |  uint8 | Dummy variable which represents an employee's status as a female. 1: Yes, 0: No                  |
| 38 |  healthcare_rep_job         | 1470 non-null |  uint8 | Dummy variable that represents job status as a healthcare representative. 1: Yes, 0: No          |
| 39 |  hr_job                     | 1470 non-null |  uint8 | Dummy variable that represents job status as a human resources worker. 1: Yes, 0: No             |
| 40 |  lab_tech_job               | 1470 non-null |  uint8 | Dummy variable that represents job status as a laboratory technician. 1: Yes, 0: No              |
| 41 |  manager_job                | 1470 non-null |  uint8 | Dummy variable that represents job status as a manager. 1: Yes, 0: No                            |
| 42 |  manufacturing_dir_job      | 1470 non-null |  uint8 | Dummy variable that represents job status as a manufacturing director. 1: Yes, 0: No             |
| 43 |  research_dir_job           | 1470 non-null |  uint8 | Dummy variable that represents job status as a research director. 1: Yes, 0: No                  |
| 44 |  research_scientist_job     | 1470 non-null |  uint8 | Dummy variable that represents job status as a research scientist. 1: Yes, 0: No                 |
| 45 |  sales_exec_job             | 1470 non-null |  uint8 | Dummy variable that represents job status as a sales executive. 1: Yes, 0: No                    |
| 46 |  sales_rep_job              | 1470 non-null |  uint8 | Dummy variable that represents job status as a sales representative. 1: Yes, 0: No               |
| 47 |  divorced                   | 1470 non-null |  uint8 | Dummy variable that represents marital status as divorced. 1: Yes, 0: No                         |
| 48 |  married                    | 1470 non-null |  uint8 | Dummy variable that represents marital status as married. 1: Ye, 0: No                           |
| 49 |  single                     | 1470 non-null |  uint8 | Dummy variable that represents marital status as single. 1: Yes , 0: No                          |