from api.model_service import ModelService


def main() -> None:
    service = ModelService.from_env()
    service.load_once()
    comments = ["I love this product!", "This is the worst experience."]
    predictions = service.predict(comments)
    print(
        [
            {"comment": comment, "sentiment": sentiment}
            for comment, sentiment in zip(comments, predictions)
        ]
    )


if __name__ == "__main__":
    main()
