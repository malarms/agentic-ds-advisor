
from app.agents.planner import create_analysis_plan


def main():
    columns = [
        "age",
        "monthly_charges",
        "contract_type",
        "churn",
    ]

    plan = create_analysis_plan(
        request=(
            "Explore the dataset, identify data-quality issues, "
            "and build a classification model to predict churn."
        ),
        columns=columns,
    )

    print(plan.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
