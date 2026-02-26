import math

import numpy as np
import pandas as pd
import streamlit as st


st.set_page_config(page_title="Chi-Square Helper", layout="wide")
st.title("Chi-Square Helper")


def _gammaincc(a: float, x: float) -> float:
	"""Regularized upper incomplete gamma Q(a, x)."""
	if a <= 0 or x < 0:
		return float("nan")
	if x == 0:
		return 1.0

	eps = 1e-14
	fpmin = 1e-300
	itmax = 200

	# Series representation for P(a, x), then Q = 1 - P
	if x < a + 1.0:
		ap = a
		summation = 1.0 / a
		delta = summation
		for _ in range(itmax):
			ap += 1.0
			delta *= x / ap
			summation += delta
			if abs(delta) < abs(summation) * eps:
				break

		p_val = (
			summation
			* math.exp(-x + a * math.log(x) - math.lgamma(a))
		)
		return max(0.0, min(1.0, 1.0 - p_val))

	# Continued fraction representation for Q(a, x)
	b = x + 1.0 - a
	c = 1.0 / fpmin
	d = 1.0 / b
	h = d

	for i in range(1, itmax + 1):
		an = -i * (i - a)
		b += 2.0
		d = an * d + b
		if abs(d) < fpmin:
			d = fpmin
		c = b + an / c
		if abs(c) < fpmin:
			c = fpmin
		d = 1.0 / d
		delta = d * c
		h *= delta
		if abs(delta - 1.0) < eps:
			break

	q_val = math.exp(-x + a * math.log(x) - math.lgamma(a)) * h
	return max(0.0, min(1.0, q_val))


def chi_square_p_value(chi_square_stat: float, dof: int) -> float:
	if chi_square_stat < 0 or dof < 1:
		return float("nan")
	return _gammaincc(dof / 2.0, chi_square_stat / 2.0)


def initialize_table(data_row_count: int, existing: pd.DataFrame | None = None) -> pd.DataFrame:
	rows = []
	for i in range(data_row_count):
		phenotype = f"Phenotype {i + 1}"
		observed = 0.0
		expected = 0.0

		if existing is not None and i < len(existing) - 1:
			phenotype = str(existing.loc[i, "Phenotype"])
			observed = pd.to_numeric(existing.loc[i, "Observed"], errors="coerce")
			expected = pd.to_numeric(existing.loc[i, "Expected"], errors="coerce")
			observed = 0.0 if pd.isna(observed) else float(observed)
			expected = 0.0 if pd.isna(expected) else float(expected)

		rows.append(
			{
				"Phenotype": phenotype,
				"Observed": observed,
				"Expected": expected,
				"O-E": np.nan,
				"(O-E)^2": np.nan,
				"(O-E)^2/E": np.nan,
			}
		)

	rows.append(
		{
			"Phenotype": "Totals",
			"Observed": 0.0,
			"Expected": 0.0,
			"O-E": np.nan,
			"(O-E)^2": np.nan,
			"(O-E)^2/E": 0.0,
		}
	)
	return pd.DataFrame(rows)


def recompute_table(df: pd.DataFrame, data_row_count: int) -> pd.DataFrame:
	out = df.copy()

	out.loc[: data_row_count - 1, "Observed"] = pd.to_numeric(
		out.loc[: data_row_count - 1, "Observed"], errors="coerce"
	).fillna(0.0)
	out.loc[: data_row_count - 1, "Expected"] = pd.to_numeric(
		out.loc[: data_row_count - 1, "Expected"], errors="coerce"
	).fillna(0.0)

	observed = out.loc[: data_row_count - 1, "Observed"].astype(float)
	expected = out.loc[: data_row_count - 1, "Expected"].astype(float)

	diff = observed - expected
	diff_sq = diff**2
	with np.errstate(divide="ignore", invalid="ignore"):
		chi_terms = np.where(expected != 0, diff_sq / expected, np.nan)

	out.loc[: data_row_count - 1, "O-E"] = diff
	out.loc[: data_row_count - 1, "(O-E)^2"] = diff_sq
	out.loc[: data_row_count - 1, "(O-E)^2/E"] = chi_terms

	total_index = data_row_count
	out.loc[total_index, "Phenotype"] = "Totals"
	out.loc[total_index, "Observed"] = float(observed.sum())
	out.loc[total_index, "Expected"] = float(expected.sum())
	out.loc[total_index, "O-E"] = np.nan
	out.loc[total_index, "(O-E)^2"] = np.nan
	out.loc[total_index, "(O-E)^2/E"] = float(np.nansum(chi_terms))

	return out


dof = st.selectbox(
	"Degrees of freedom",
	options=list(range(1, 10)),
	index=0,
	help="For chi-square goodness-of-fit, this creates df + 1 phenotype rows.",
)

data_rows = dof + 1

if "chi_table" not in st.session_state or st.session_state.get("data_rows") != data_rows:
	existing_table = st.session_state.get("chi_table")
	st.session_state.chi_table = initialize_table(data_rows, existing=existing_table)
	st.session_state.data_rows = data_rows

current_table = recompute_table(st.session_state.chi_table, data_rows)

edited_table = st.data_editor(
	current_table,
	hide_index=True,
	num_rows="fixed",
	use_container_width=True,
	disabled=["O-E", "(O-E)^2", "(O-E)^2/E"],
	column_config={
		"Phenotype": st.column_config.TextColumn(help="Label only; not used in calculations."),
		"Observed": st.column_config.NumberColumn(format="%.4f"),
		"Expected": st.column_config.NumberColumn(format="%.4f"),
		"O-E": st.column_config.NumberColumn(format="%.4f"),
		"(O-E)^2": st.column_config.NumberColumn(format="%.4f"),
		"(O-E)^2/E": st.column_config.NumberColumn(format="%.6f"),
	},
)

# Persist only editable cells for phenotype rows. Totals are always recalculated.
persist = st.session_state.chi_table.copy()
persist.loc[: data_rows - 1, ["Phenotype", "Observed", "Expected"]] = edited_table.loc[
	: data_rows - 1, ["Phenotype", "Observed", "Expected"]
].values
st.session_state.chi_table = recompute_table(persist, data_rows)

chi_square_stat = float(st.session_state.chi_table.loc[data_rows, "(O-E)^2/E"])
p_value = chi_square_p_value(chi_square_stat, dof)

col1, col2 = st.columns(2)
col1.metric("Chi-square statistic", f"{chi_square_stat:.6f}")
col2.metric("p-value", f"{p_value:.6f}")

st.caption("Note: Expected values of 0 make (O-E)^2/E undefined for that row and are skipped in the total chi-square sum.")
