
CREATE TABLE analytical_schema.dim_date (
	date_key NUMERIC(38, 0), 
	calendar_date TIMESTAMP WITHOUT TIME ZONE, 
	calendar_year NUMERIC(4, 0), 
	quarter NUMERIC(2, 0), 
	year_quarter_num NUMERIC(38, 0), 
	month_num NUMERIC(2, 0), 
	year_month_num NUMERIC(38, 0), 
	month_name TEXT, 
	month_short_name VARCHAR(7), 
	month_year TEXT, 
	calendar_day NUMERIC(2, 0), 
	day_of_week NUMERIC(2, 0), 
	day_name VARCHAR(7), 
	week_of_year NUMERIC(2, 0), 
	day_of_year NUMERIC(4, 0), 
	year_dayofyear NUMERIC(38, 0), 
	holiday_ind VARCHAR(7), 
	holiday_desc TEXT, 
	weekend_ind VARCHAR(7), 
	business_day NUMERIC(38, 0), 
	fiscal_year NUMERIC(5, 0), 
	fiscal_quarter NUMERIC(10, 0), 
	fiscal_month_short_name VARCHAR(7), 
	fiscal_month NUMERIC(2, 0), 
	fiscal_month_year TEXT, 
	fiscal_week NUMERIC(2, 0), 
	fiscal_year_month_num NUMERIC(38, 0), 
	relative_month_num NUMERIC(18, 0), 
	relative_fiscal_week_num NUMERIC(18, 0), 
	relative_week_num NUMERIC(18, 0), 
	relative_fiscal_month_num NUMERIC(18, 0), 
	etl_inserted_invocation_id VARCHAR(128), 
	etl_inserted_date VARCHAR(26)
)

/*
3 rows from dim_date table:
date_key	calendar_date	calendar_year	quarter	year_quarter_num	month_num	year_month_num	month_name	month_short_name	month_year	calendar_day	day_of_week	day_name	week_of_year	day_of_year	year_dayofyear	holiday_ind	holiday_desc	weekend_ind	business_day	fiscal_year	fiscal_quarter	fiscal_month_short_name	fiscal_month	fiscal_month_year	fiscal_week	fiscal_year_month_num	relative_month_num	relative_fiscal_week_num	relative_week_num	relative_fiscal_month_num	etl_inserted_invocation_id	etl_inserted_date
20200101	2020-01-01 00:00:00	2020	1	20201	1	202001	January  	Jan	Jan 2020	1	4	Wed	1	1	2020001	Y	New Year's Day	N	0	2020	3	Jan	7	Jan 2020	1	202007	1	1	1	1	SYNTHETIC_LOAD	2025-04-08 12:18:28
20200102	2020-01-02 00:00:00	2020	1	20201	1	202001	January  	Jan	Jan 2020	2	5	Thu	1	2	2020002	N	None	N	1	2020	3	Jan	7	Jan 2020	1	202007	1	1	1	1	SYNTHETIC_LOAD	2025-04-08 12:18:28
20200103	2020-01-03 00:00:00	2020	1	20201	1	202001	January  	Jan	Jan 2020	3	6	Fri	1	3	2020003	N	None	N	1	2020	3	Jan	7	Jan 2020	1	202007	1	1	1	1	SYNTHETIC_LOAD	2025-04-08 12:18:28
*/


CREATE TABLE analytical_schema.dim_ukg_company_details (
	company_details_dim_id INTEGER DEFAULT nextval('analytical_schema.dim_ukg_company_details_company_details_dim_id_seq'::regclass) NOT NULL, 
	is_master_company BOOLEAN, 
	master_company_id TEXT, 
	company_id TEXT NOT NULL, 
	company_code TEXT NOT NULL, 
	company_doing_business_as_name TEXT, 
	company_gl_segment TEXT, 
	company_name TEXT, 
	tax_calculation_group_id TEXT, 
	contract_number TEXT, 
	address_line1 TEXT, 
	address_line2 TEXT, 
	address_city TEXT, 
	address_state TEXT, 
	address_zip_code TEXT, 
	address_country TEXT, 
	address_county TEXT, 
	phone_number TEXT, 
	phone_number_extension TEXT, 
	federal_tax_id NUMERIC(38, 0) NOT NULL, 
	other_federal_tax_id TEXT, 
	organization_level1_label TEXT, 
	organization_level2_label TEXT, 
	organization_level3_label TEXT, 
	organization_level4_label TEXT, 
	currency_code TEXT, 
	date_of_business_closure TEXT, 
	use_position_management BOOLEAN, 
	use_multiple_job_groups BOOLEAN, 
	integration_record_id TEXT, 
	first_observed_date TIMESTAMP WITHOUT TIME ZONE, 
	last_observed_date TIMESTAMP WITHOUT TIME ZONE, 
	hex_encoding TEXT, 
	CONSTRAINT dim_ukg_company_details_pkey PRIMARY KEY (company_id, company_code, federal_tax_id)
)

/*
3 rows from dim_ukg_company_details table:
company_details_dim_id	is_master_company	master_company_id	company_id	company_code	company_doing_business_as_name	company_gl_segment	company_name	tax_calculation_group_id	contract_number	address_line1	address_line2	address_city	address_state	address_zip_code	address_country	address_county	phone_number	phone_number_extension	federal_tax_id	other_federal_tax_id	organization_level1_label	organization_level2_label	organization_level3_label	organization_level4_label	currency_code	date_of_business_closure	use_position_management	use_multiple_job_groups	integration_record_id	first_observed_date	last_observed_date	hex_encoding
1	False	MC-2	COMP-1	CC001	DBA-c4ca4238	Manufacturing	Company 1 Inc.	TCG-1	CNTR-743	5870 Main St	None	New York	AZ	46629	USA	Kings	(182) 897-3507	None	272605501	801527985	Division B	Region 1	Department D	Team 8	USD	None	True	True	c4ca4238a0b923820dcc509a6f75849b	2018-12-24 00:00:00	2025-04-08 00:00:00	c4ca4238a0b92382
2	True	None	COMP-2	CC002	DBA-c81e728d	Healthcare	Company 2 Inc.	TCG-1	None	7824 Main St	None	New York	NY	88055	USA	None	(663) 381-5275	None	953609463	None	Division B	Region 6	Department A	None	USD	None	False	False	c81e728d9d4c2f636f067f89cc14862c	2022-04-07 00:00:00	2025-04-08 00:00:00	c81e728d9d4c2f63
3	False	MC-1	COMP-3	CC003	DBA-eccbc87e	Finance	Company 3 Inc.	TCG-4	CNTR-524	5654 Main St	None	Phoenix	NY	72129	USA	Cook	(290) 953-9194	235	215025040	None	Division E	Region 10	Department D	Team 9	USD	None	False	True	eccbc87e4b5ce2fe28308fd9f2a7baf3	2025-02-12 00:00:00	2025-04-08 00:00:00	eccbc87e4b5ce2fe
*/


CREATE TABLE analytical_schema.dim_ukg_employee_demographic_details (
	employee_demographic_dim_id INTEGER DEFAULT nextval('analytical_schema.dim_ukg_employee_demographic_de_employee_demographic_dim_id_seq'::regclass) NOT NULL, 
	ethnic_description TEXT, 
	additional_name1 TEXT, 
	additional_name2 TEXT, 
	address_id TEXT, 
	address_is_on_tax_boundary TEXT, 
	address_latitude DOUBLE PRECISION, 
	disability_type TEXT, 
	former_name TEXT, 
	health_blood_type TEXT, 
	health_eyes TEXT, 
	health_hair TEXT, 
	health_height_feet TEXT, 
	health_height_inches TEXT, 
	health_last_donate_date TIMESTAMP WITHOUT TIME ZONE, 
	health_weight DOUBLE PRECISION, 
	previous_ssn TEXT, 
	origin_country TEXT, 
	origin_location TEXT, 
	address_sms TEXT, 
	home_phone_is_private BOOLEAN, 
	last_name_not_same_as_ss_card TEXT, 
	nationality1 TEXT, 
	nationality2 TEXT, 
	nationality3 TEXT, 
	person_id TEXT, 
	employee_id TEXT NOT NULL, 
	company_id TEXT, 
	first_name TEXT, 
	middle_name TEXT, 
	last_name TEXT, 
	preferred_name TEXT, 
	name_prefix_code TEXT, 
	name_suffix_code TEXT, 
	email_address TEXT, 
	email_address_alternate TEXT, 
	home_phone_id TEXT, 
	home_phone TEXT, 
	home_phone_country TEXT, 
	address_line1 TEXT, 
	address_line2 TEXT, 
	address_line3 TEXT, 
	address_line4 TEXT, 
	address_city TEXT, 
	address_state TEXT, 
	address_zip_code TEXT NOT NULL, 
	address_country TEXT, 
	address_county TEXT, 
	date_of_birth TIMESTAMP WITHOUT TIME ZONE, 
	gender TEXT, 
	ethnic_id_code TEXT, 
	is_smoker BOOLEAN, 
	is_disabled TEXT, 
	marital_status_code TEXT, 
	ssn TEXT, 
	ssn_is_suppressed BOOLEAN, 
	user_id TEXT, 
	integration_record_id TEXT, 
	cell_phone_number TEXT, 
	first_observed_date TIMESTAMP WITHOUT TIME ZONE, 
	last_observed_date TIMESTAMP WITHOUT TIME ZONE, 
	hex_encoding TEXT, 
	CONSTRAINT dim_ukg_employee_demographic_details_pkey PRIMARY KEY (employee_id, address_zip_code)
)

/*
3 rows from dim_ukg_employee_demographic_details table:
employee_demographic_dim_id	ethnic_description	additional_name1	additional_name2	address_id	address_is_on_tax_boundary	address_latitude	disability_type	former_name	health_blood_type	health_eyes	health_hair	health_height_feet	health_height_inches	health_last_donate_date	health_weight	previous_ssn	origin_country	origin_location	address_sms	home_phone_is_private	last_name_not_same_as_ss_card	nationality1	nationality2	nationality3	person_id	employee_id	company_id	first_name	middle_name	last_name	preferred_name	name_prefix_code	name_suffix_code	email_address	email_address_alternate	home_phone_id	home_phone	home_phone_country	address_line1	address_line2	address_line3	address_line4	address_city	address_state	address_zip_code	address_country	address_county	date_of_birth	gender	ethnic_id_code	is_smoker	is_disabled	marital_status_code	ssn	ssn_is_suppressed	user_id	integration_record_id	cell_phone_number	first_observed_date	last_observed_date	hex_encoding
1	None	None	None	ADDR-1	None	44.93704484188748	None	None	None	Hazel	None	None	0	None	153.32248388227373	None	USA	None	N	True	N	American	None	None	PERS-1	EMP-000001	COMP-5	Nancy	None	Garcia	Joseph	None	None	charles.garcia@company.com	None	PH-1	(511) 440-6238	USA	6355 Lake St	None	None	None	San Antonio	IL	10011	USA	None	1979-01-15 00:00:00	Female	BLK	False	None	Single	858367614	False	USER-1	c4ca4238a0b923820dcc509a6f75849b	(578) 412-8871	2023-01-31 00:00:00	2025-04-08 00:00:00	fd8c0e0484a6f50e
2	White	AKA Christopher	None	ADDR-2	Y	41.85910754911218	None	None	B-	None	None	4	None	None	None	None	USA	None	Y	False	N	None	None	None	PERS-2	EMP-000002	COMP-7	Nancy	P.	Garcia	Joseph	None	II	charles.garcia@company.com	None	PH-2	(984) 170-2612	USA	7095 Maple Ct	Apt 443	None	None	New York	TX	10021	USA	None	1983-12-02 00:00:00	Female	ASN	False	N	Divorced	977522135	False	USER-2	c81e728d9d4c2f636f067f89cc14862c	(528) 619-1680	2018-12-30 00:00:00	2025-04-08 00:00:00	0536bbd99375e5b9
3	White	None	None	ADDR-3	None	41.22778369708232	None	Martinez	None	Blue	None	None	None	None	100.54892929242372	None	India	None	Y	True	N	American	Indian	None	PERS-3	EMP-000003	COMP-10	Nancy	None	Garcia	Joseph	Dr.	III	charles.garcia@company.com	richard.thomas@personal.com	PH-3	(274) 744-4878	USA	2457 Lake Dr	None	None	None	San Jose	TX	10030	USA	Santa Clara	1978-07-10 00:00:00	Male	HIS	True	N	Divorced	602413848	False	USER-3	eccbc87e4b5ce2fe28308fd9f2a7baf3	(591) 807-2568	2016-10-30 00:00:00	2025-04-08 00:00:00	963a005b8979036d
*/


CREATE TABLE analytical_schema.dim_ukg_employee_education (
	"employeeEducationDimId" SERIAL NOT NULL, 
	"employeeId" VARCHAR(255) NOT NULL, 
	"systemId" VARCHAR(255) NOT NULL, 
	school VARCHAR(255), 
	"educationLevel" VARCHAR(255), 
	"educationMajor" VARCHAR(255), 
	"educationMinor" VARCHAR(255), 
	gpa DOUBLE PRECISION, 
	"beginDate" TIMESTAMP WITHOUT TIME ZONE NOT NULL, 
	"endDate" TIMESTAMP WITHOUT TIME ZONE NOT NULL, 
	"isGraduate" BOOLEAN, 
	"isHighestLevel" BOOLEAN, 
	"employeeNumber" VARCHAR(255), 
	country VARCHAR(255), 
	"firstObservedDate" TIMESTAMP WITHOUT TIME ZONE, 
	"lastObservedDate" TIMESTAMP WITHOUT TIME ZONE, 
	"hexEncoding" VARCHAR(255), 
	CONSTRAINT dim_ukg_employee_education_pkey PRIMARY KEY ("employeeEducationDimId")
)

/*
3 rows from dim_ukg_employee_education table:
employeeEducationDimId	employeeId	systemId	school	educationLevel	educationMajor	educationMinor	gpa	beginDate	endDate	isGraduate	isHighestLevel	employeeNumber	country	firstObservedDate	lastObservedDate	hexEncoding
1	EMP755651	SYS550846	Stanford University	Master	Computer Science	Mathematics	3.42	2015-07-07 16:18:11.174400	2023-04-10 20:59:34.166400	True	False	EMP617007	USA	2020-04-15 19:04:40.863605	2020-06-05 09:41:27.577205	4637ef6ddf9ecf9fe7338b516701883e910efa6d9d18b48f758226249f542b08
2	EMP470586	SYS217487	MIT	Bachelor	Chemical Engineering	Mathematics	3.56	2018-03-03 16:53:53.116800	2021-11-24 22:09:09.792000	True	False	EMP204691	Australia	2021-01-23 03:17:52.258805	2021-03-23 05:08:56.377205	73e0a7b583ab4f55ecf672f21f0db1b1ece7aa51a05922371f75719d989580a9
3	EMP801160	SYS140754	California Institute of Technology (Caltech)	Bachelor	Mechanical Engineering	Mathematics	0.81	2021-02-13 09:37:46.099200	2023-02-24 13:58:31.065600	True	True	EMP502930	USA	2021-08-14 09:21:27.049205	2023-01-06 10:18:03.346805	0601bd7f38cbe62b46ed10293baa80d84c88afdc0fa4a7341dc9c2953b05be1e
*/


CREATE TABLE analytical_schema.dim_ukg_employee_job_history_details (
	employee_job_history_dim_id INTEGER DEFAULT nextval('analytical_schema.dim_ukg_employee_job_history_de_employee_job_history_dim_id_seq'::regclass) NOT NULL, 
	annual_salary DOUBLE PRECISION, 
	company_id VARCHAR(50) NOT NULL, 
	date_time_created TIMESTAMP WITHOUT TIME ZONE, 
	employee_id VARCHAR(50) NOT NULL, 
	employee_type VARCHAR(50), 
	employee_status VARCHAR(50), 
	flsa_category VARCHAR(50), 
	full_time_or_part_time VARCHAR(50), 
	hourly_pay_rate DOUBLE PRECISION, 
	is_job_change BOOLEAN, 
	is_orgchange BOOLEAN, 
	is_outside_guidelines BOOLEAN, 
	is_outside_range BOOLEAN, 
	is_promotion BOOLEAN, 
	is_rate_change BOOLEAN, 
	is_system BOOLEAN, 
	is_transfer BOOLEAN, 
	job_code VARCHAR(50), 
	job_description VARCHAR(100), 
	job_effective_date TIMESTAMP WITHOUT TIME ZONE NOT NULL, 
	job_group_code VARCHAR(50), 
	location_code VARCHAR(50), 
	org_level1_code VARCHAR(50), 
	org_level2_code VARCHAR(50), 
	org_level3_code VARCHAR(50), 
	org_level4_code VARCHAR(50), 
	other_rate1 DOUBLE PRECISION, 
	other_rate2 DOUBLE PRECISION, 
	other_rate3 DOUBLE PRECISION, 
	other_rate4 DOUBLE PRECISION, 
	pay_group_code VARCHAR(50), 
	pay_period_code VARCHAR(50), 
	pay_scale_code VARCHAR(50), 
	percent_change DOUBLE PRECISION, 
	period_pay_rate DOUBLE PRECISION, 
	piece_pay_rate DOUBLE PRECISION, 
	position_code VARCHAR(50), 
	reason_code VARCHAR(50) NOT NULL, 
	salary_grade VARCHAR(50), 
	salary_or_hourly VARCHAR(50), 
	scheduled_annual_hours DOUBLE PRECISION, 
	scheduled_full_time_equivalency DOUBLE PRECISION, 
	scheduled_work_hours DOUBLE PRECISION, 
	shift_code VARCHAR(50), 
	shift_group_code VARCHAR(50), 
	step_number DOUBLE PRECISION, 
	supervisor_id VARCHAR(50), 
	supervisor_name_first VARCHAR(50), 
	supervisor_name_last VARCHAR(50), 
	supervisor_name_suffix VARCHAR(50), 
	supervisor_not_in_list BOOLEAN, 
	system_id VARCHAR(50), 
	union_national VARCHAR(50), 
	union_local VARCHAR(50), 
	use_pay_scales BOOLEAN, 
	weekly_pay_rate DOUBLE PRECISION, 
	notes VARCHAR(500), 
	home_company_id VARCHAR(50), 
	integration_effective_date TIMESTAMP WITHOUT TIME ZONE, 
	project_code VARCHAR(50), 
	number_of_payments DOUBLE PRECISION, 
	weekly_hours DOUBLE PRECISION, 
	is_viewable_by_employee BOOLEAN, 
	created_by_user_id DOUBLE PRECISION, 
	job_title VARCHAR(100), 
	first_observed_date TIMESTAMP WITHOUT TIME ZONE, 
	last_observed_date TIMESTAMP WITHOUT TIME ZONE, 
	hex_encoding VARCHAR(100), 
	CONSTRAINT pk_ukg_employee_job_history_details PRIMARY KEY (company_id, employee_id, job_effective_date, reason_code)
)

/*
3 rows from dim_ukg_employee_job_history_details table:
employee_job_history_dim_id	annual_salary	company_id	date_time_created	employee_id	employee_type	employee_status	flsa_category	full_time_or_part_time	hourly_pay_rate	is_job_change	is_orgchange	is_outside_guidelines	is_outside_range	is_promotion	is_rate_change	is_system	is_transfer	job_code	job_description	job_effective_date	job_group_code	location_code	org_level1_code	org_level2_code	org_level3_code	org_level4_code	other_rate1	other_rate2	other_rate3	other_rate4	pay_group_code	pay_period_code	pay_scale_code	percent_change	period_pay_rate	piece_pay_rate	position_code	reason_code	salary_grade	salary_or_hourly	scheduled_annual_hours	scheduled_full_time_equivalency	scheduled_work_hours	shift_code	shift_group_code	step_number	supervisor_id	supervisor_name_first	supervisor_name_last	supervisor_name_suffix	supervisor_not_in_list	system_id	union_national	union_local	use_pay_scales	weekly_pay_rate	notes	home_company_id	integration_effective_date	project_code	number_of_payments	weekly_hours	is_viewable_by_employee	created_by_user_id	job_title	first_observed_date	last_observed_date	hex_encoding
1	None	COMP001	2020-07-01 04:14:05	EMP0001C01	PART_TIME	ACTIVE	NON_EXEMPT	PART_TIME	None	True	False	False	False	False	False	True	False	MGR	Job description for EMP0001C01	2020-07-01 04:14:05	None	LOC6	DEPT1	None	None	None	None	None	None	None	PG2	BIWEEKLY	None	None	1479.4958927940777	1.2005923155123401	POS31	HIRE	None	HOURLY	None	None	45.855868763297686	SHIFT3	None	None	SUP28	First18	Last17	None	False	SYS2506	UN7	None	True	None	Note about employee EMP0001C01	None	None	None	None	41.46384321399975	False	5824.0	HR Specialist	2020-07-01 04:14:05	2025-04-08 14:30:20.269532	None
2	140367.19268431613	COMP001	2019-08-28 07:09:12	EMP0001C01	TEMPORARY	LOA	NON_EXEMPT	PART_TIME	46.86653198693065	False	False	False	False	False	False	False	False	HR	Job description for EMP0001C01	2019-08-28 07:09:12	None	LOC10	DEPT8	TEAM17	None	UNIT7	None	None	32.70875531132056	None	PG4	SEMIMONTHLY	None	None	3720.0284023777135	None	POS13	SAL_ADJ	SG2	HOURLY	None	None	57.953892909551584	SHIFT5	None	3.0	SUP45	First7	None	None	True	None	None	None	True	None	Note about employee EMP0001C01	None	None	None	None	30.721379651610896	True	None	Operations	2019-08-28 07:09:12	2025-04-08 14:30:20.269532	None
3	None	COMP001	2016-11-01 11:40:20	EMP0002C01	FULL_TIME	LOA	EXEMPT	PART_TIME	None	False	False	False	False	False	False	False	False	MKTG	Job description for EMP0002C01	2016-11-01 11:40:20	None	LOC14	DEPT4	TEAM4	None	None	None	None	None	65.29745718471753	PG2	SEMIMONTHLY	None	1.420544932414316	505.5753255126064	None	POS24	STATUS_CHG	None	HOURLY	None	0.8705548995344343	None	SHIFT2	None	7.0	SUP14	None	Last13	Jr	False	None	UN2	UL6	False	662.7855273138487	None	None	None	None	None	32.35627919181057	False	8816.0	Software Developer	2016-11-01 11:40:20	2025-04-08 14:30:20.269532	None
*/


CREATE TABLE analytical_schema.dim_ukg_org_levels (
	org_levels_dim_id INTEGER DEFAULT nextval('analytical_schema.dim_ukg_org_levels_org_levels_dim_id_seq'::regclass) NOT NULL, 
	budget_group VARCHAR(100), 
	code VARCHAR(50) NOT NULL, 
	current_year_budget_fte DOUBLE PRECISION, 
	current_year_budget_salary DOUBLE PRECISION, 
	description VARCHAR(200), 
	gl_segment VARCHAR(50), 
	is_active BOOLEAN, 
	last_year_budget_fte DOUBLE PRECISION, 
	last_year_budget_salary DOUBLE PRECISION, 
	level INTEGER, 
	level_description VARCHAR(100), 
	reporting_category VARCHAR(50) NOT NULL, 
	first_observed_date TIMESTAMP WITHOUT TIME ZONE, 
	last_observed_date TIMESTAMP WITHOUT TIME ZONE, 
	hex_encoding VARCHAR(100), 
	CONSTRAINT pk_ukg_org_levels PRIMARY KEY (code, reporting_category)
)

/*
3 rows from dim_ukg_org_levels table:
org_levels_dim_id	budget_group	code	current_year_budget_fte	current_year_budget_salary	description	gl_segment	is_active	last_year_budget_fte	last_year_budget_salary	level	level_description	reporting_category	first_observed_date	last_observed_date	hex_encoding
1	R&D	DEPT001	17.67263081439807	650908.9281937131	Department 1 Description	GL1001	True	20.056420696593666	456700.16563316726	1	Department	HR	2020-10-12 18:55:07.305600	2025-04-08 14:35:57.800704	None
2	SALES	DEPT002	42.91822160756673	329132.91974314355	Department 2 Description	GL1002	True	39.943706457004694	686484.1970728624	1	Department	FINANCE	2020-09-28 04:25:44.428800	2025-04-08 14:35:57.800704	None
3	ADMIN	DEPT003	46.83795398265128	920336.3421495848	Department 3 Description	GL1003	True	41.6666824341661	720592.7831932531	1	Department	OPERATIONS	2021-06-04 03:30:01.785600	2025-04-08 14:35:57.800704	None
*/


CREATE TABLE analytical_schema.dim_ukg_person_details (
	person_details_dim_id INTEGER DEFAULT nextval('analytical_schema.dim_ukg_person_details_person_details_dim_id_seq'::regclass) NOT NULL, 
	additional_name1 VARCHAR(100), 
	additional_name2 VARCHAR(100), 
	address_id VARCHAR(50) NOT NULL, 
	address_is_on_tax_boundary BOOLEAN, 
	address_latitude DOUBLE PRECISION, 
	cobra_export VARCHAR(50), 
	cobra_is_active BOOLEAN, 
	cobra_reason VARCHAR(50), 
	cobra_status VARCHAR(50), 
	cobra_status_date TIMESTAMP WITHOUT TIME ZONE, 
	community_broadcast_sms_code VARCHAR(50), 
	consent_electronic_w2 BOOLEAN, 
	consent_electronic_w2pr BOOLEAN, 
	date_deceased TIMESTAMP WITHOUT TIME ZONE, 
	date_of_cobra_event TIMESTAMP WITHOUT TIME ZONE, 
	date_of_cobra_export TIMESTAMP WITHOUT TIME ZONE, 
	date_of_cobra_letter TIMESTAMP WITHOUT TIME ZONE, 
	date_of_i9_expiration TIMESTAMP WITHOUT TIME ZONE, 
	datetime_changed TIMESTAMP WITHOUT TIME ZONE NOT NULL, 
	datetime_created TIMESTAMP WITHOUT TIME ZONE, 
	disability_type VARCHAR(50), 
	ethnic_description VARCHAR(100), 
	former_name VARCHAR(100), 
	health_blood_type VARCHAR(10), 
	health_eyes VARCHAR(20), 
	health_hair VARCHAR(20), 
	health_height_feet VARCHAR(5), 
	health_height_inches VARCHAR(5), 
	health_last_donate_date TIMESTAMP WITHOUT TIME ZONE, 
	health_weight DOUBLE PRECISION, 
	i9_alien_number VARCHAR(50), 
	i9_doc_a VARCHAR(50), 
	i9_doc_b VARCHAR(50), 
	i9_doc_c VARCHAR(50), 
	i9_verified BOOLEAN, 
	i9_visa_type VARCHAR(50), 
	i9_work_auth VARCHAR(50), 
	is_disabled BOOLEAN, 
	is_multi_pay_group BOOLEAN, 
	is_smoker BOOLEAN, 
	military_service BOOLEAN, 
	military_branch_served VARCHAR(50), 
	military_era VARCHAR(50), 
	military_is_disabled_vet VARCHAR(50), 
	military_is_oth_elig_vet VARCHAR(50), 
	military_is_oth_elig_vet_basis VARCHAR(50), 
	military_is_active_wartime_vet VARCHAR(50), 
	name_former VARCHAR(100), 
	previous_ssn VARCHAR(20), 
	origin_country VARCHAR(50), 
	origin_location VARCHAR(100), 
	w2_is_deceased BOOLEAN, 
	cobra_notes TEXT, 
	address_sms VARCHAR(50), 
	military_separation_date TIMESTAMP WITHOUT TIME ZONE, 
	home_phone_is_private BOOLEAN, 
	sms_approvals BOOLEAN, 
	sms_pay_notification BOOLEAN, 
	i9_visa_expiration_date TIMESTAMP WITHOUT TIME ZONE, 
	military_is_medal_vet VARCHAR(50), 
	last_name_not_same_as_sscard VARCHAR(100), 
	chk_cashing_inst_code VARCHAR(50), 
	nationality1 VARCHAR(50), 
	nationality2 VARCHAR(50), 
	nationality3 VARCHAR(50), 
	person_id VARCHAR(50) NOT NULL, 
	employee_id VARCHAR(50) NOT NULL, 
	company_id VARCHAR(50), 
	user_name VARCHAR(50), 
	first_name VARCHAR(50), 
	middle_name VARCHAR(50), 
	last_name VARCHAR(50), 
	preferred_name VARCHAR(50), 
	name_prefix_code VARCHAR(10), 
	name_suffix_code VARCHAR(10), 
	email_address VARCHAR(100), 
	email_address_alternate VARCHAR(100), 
	home_phone VARCHAR(20), 
	home_phone_country VARCHAR(10), 
	address_line1 VARCHAR(100), 
	address_line2 VARCHAR(100), 
	address_line3 VARCHAR(100), 
	address_line4 VARCHAR(100), 
	address_city VARCHAR(50), 
	address_state VARCHAR(50), 
	address_zip_code VARCHAR(20), 
	address_country VARCHAR(50), 
	address_county VARCHAR(50), 
	date_of_birth TIMESTAMP WITHOUT TIME ZONE, 
	gender VARCHAR(10), 
	ethnic_id_code VARCHAR(50), 
	marital_status_code VARCHAR(20), 
	ssn VARCHAR(20), 
	national_id VARCHAR(50), 
	national_id_country VARCHAR(50), 
	ssn_is_suppressed BOOLEAN, 
	first_observed_date TIMESTAMP WITHOUT TIME ZONE, 
	last_observed_date TIMESTAMP WITHOUT TIME ZONE, 
	hex_encoding VARCHAR(100), 
	CONSTRAINT pk_ukg_person_details PRIMARY KEY (person_id, employee_id, address_id, datetime_changed)
)

/*
3 rows from dim_ukg_person_details table:
person_details_dim_id	additional_name1	additional_name2	address_id	address_is_on_tax_boundary	address_latitude	cobra_export	cobra_is_active	cobra_reason	cobra_status	cobra_status_date	community_broadcast_sms_code	consent_electronic_w2	consent_electronic_w2pr	date_deceased	date_of_cobra_event	date_of_cobra_export	date_of_cobra_letter	date_of_i9_expiration	datetime_changed	datetime_created	disability_type	ethnic_description	former_name	health_blood_type	health_eyes	health_hair	health_height_feet	health_height_inches	health_last_donate_date	health_weight	i9_alien_number	i9_doc_a	i9_doc_b	i9_doc_c	i9_verified	i9_visa_type	i9_work_auth	is_disabled	is_multi_pay_group	is_smoker	military_service	military_branch_served	military_era	military_is_disabled_vet	military_is_oth_elig_vet	military_is_oth_elig_vet_basis	military_is_active_wartime_vet	name_former	previous_ssn	origin_country	origin_location	w2_is_deceased	cobra_notes	address_sms	military_separation_date	home_phone_is_private	sms_approvals	sms_pay_notification	i9_visa_expiration_date	military_is_medal_vet	last_name_not_same_as_sscard	chk_cashing_inst_code	nationality1	nationality2	nationality3	person_id	employee_id	company_id	user_name	first_name	middle_name	last_name	preferred_name	name_prefix_code	name_suffix_code	email_address	email_address_alternate	home_phone	home_phone_country	address_line1	address_line2	address_line3	address_line4	address_city	address_state	address_zip_code	address_country	address_county	date_of_birth	gender	ethnic_id_code	marital_status_code	ssn	national_id	national_id_country	ssn_is_suppressed	first_observed_date	last_observed_date	hex_encoding
1	None	None	ADDR000001	True	None	None	False	None	None	None	None	True	False	None	None	None	None	None	2023-10-13 17:31:31.768713	2023-09-16 00:39:41.562380	PHYSICAL	Not Hispanic or Latino	None	AB-	None	None	6	0	2023-10-25 14:21:59.040000	169.30624778277814	None	None	DRIVERS_LICENSE	None	True	H1B	None	False	False	False	True	None	None	NO	NO	None	NO	None	XXX-XX-5019	None	None	False	COBRA notes for employee EMP0001	None	None	False	False	False	None	NO	NO	None	None	None	United Kingdom	PERS000001	EMP0001	COMP004	psmith	Patricia	Robert	Johnson	None	None	None	james.williams@example.com	None	(385) 225-5955	US	617 Main St	None	None	None	Houston	NY	90821	US	None	1978-02-10 00:00:00	FEMALE	BLACK	MARRIED	XXX-XX-5135	None	None	False	2023-09-16 07:11:14.155278	2023-10-13 17:31:31.768713	None
2	None	None	ADDR000002	False	None	None	False	None	ACTIVE	None	None	False	True	None	2025-01-06 04:09:14.457600	None	None	None	2023-10-17 13:57:47.418863	2023-10-03 01:23:26.202554	None	Not Hispanic or Latino	Williams	AB-	None	None	6	5	None	114.11898726457959	None	None	None	None	True	None	None	False	False	False	True	NAVY	None	NO	NO	None	NO	Miller	None	None	None	False	None	None	2019-10-05 05:44:32.409600	False	False	True	None	NO	NO	None	None	None	None	PERS000002	EMP0002	COMP001	jdavis	Michael	Patricia	Davis	James	Mr	None	robert.williams@example.com	None	(286) 705-2355	US	311 Main St	Apt 35	None	None	Chicago	CA	63659	US	Queens County	1994-08-24 00:00:00	FEMALE	BLACK	DIVORCED	XXX-XX-9927	ID-110104	None	False	2023-10-09 09:27:40.817276	2023-10-17 13:57:47.418863	None
3	Nickname	None	ADDR000003	True	None	EXP679	True	None	None	None	None	True	True	None	None	None	2024-10-23 16:47:46.867200	2026-12-03 15:46:48.086400	2023-06-21 05:36:07.446583	2023-06-01 08:11:15.969894	None	Not Hispanic or Latino	None	O+	None	None	6	4	None	101.76239213362832	None	None	None	None	True	H1B	AUTHORIZED	False	False	True	True	None	None	NO	NO	None	NO	None	None	None	None	False	None	(764) 598-8908	None	False	False	False	None	NO	NO	None	United States	None	None	PERS000003	EMP0003	COMP005	pjones	Michael	None	Jones	Linda	None	None	james.jones@example.com	None	(335) 715-5869	US	601 Main St	Apt 10	None	None	Chicago	CA	92879	US	None	1961-01-20 00:00:00	MALE	BLACK	WIDOWED	XXX-XX-1728	ID-850216	None	True	2023-05-25 23:22:48.958075	2023-06-21 05:36:07.446583	None
*/


CREATE TABLE analytical_schema.fact_ukg_employee_job_history_details (
	employee_id VARCHAR(50) NOT NULL, 
	job_effective_date TIMESTAMP WITHOUT TIME ZONE NOT NULL, 
	reason_code VARCHAR(50) NOT NULL, 
	employee_head_count_fytd INTEGER, 
	number_of_hires INTEGER, 
	number_of_interns INTEGER, 
	number_of_terminations INTEGER, 
	turnover DOUBLE PRECISION, 
	CONSTRAINT pk_fact_ukg_employee_job_history_details PRIMARY KEY (employee_id, job_effective_date, reason_code)
)

/*
3 rows from fact_ukg_employee_job_history_details table:
employee_id	job_effective_date	reason_code	employee_head_count_fytd	number_of_hires	number_of_interns	number_of_terminations	turnover
EMP0942	2021-10-09 19:26:04	SAL_ADJ	43	0	0	0	None
EMP0207	2021-08-27 05:51:54	TERM	90	0	0	1	0.6482723334854823
EMP0031	2024-12-14 17:49:06	TRANSFER	211	0	0	0	None
*/