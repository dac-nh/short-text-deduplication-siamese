def load_triplet_siamese_model(triplet_model_path, embedding_index, embedding_dim, lr=0.01, margin=0.4, device=torch.device("cuda:0")):
    # ---- Load triplet siamese model and distance
    triplet_model_path = "/data/dac/dedupe-project/new/model/triplet_siamese_50d_bi_gru_random"

    model = TripletSiameseModel(
        embedding_dim=[len(embedding_index), embedding_dim],
        layers=1,
        hid_dim=50,
        n_classes=30,
        bidirectional=True
    ).to(device)
    distance = TripletDistance(margin=margin).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    print(model, distance)

    # Load model and optimizer
    checkpoint = torch.load(triplet_model_path, map_location=device)
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    
    return model, distance, optimizer