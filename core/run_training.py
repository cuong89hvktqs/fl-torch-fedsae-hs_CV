from matplotlib.backends.backend_pdf import PdfPages

def run_machine_learning(server, clients, poisoned_workers, args):
    pdf_path = f"logs/{args.dataset}/{args.model_type}/{args.num_multi_class_clients}/{args.model_type}_visualize_params.pdf"
    pdf_writer = PdfPages(pdf_path)

    for epoch in range(1, args.epochs + 1):
        no_client_training = server.train_on_clients(epoch, clients, poisoned_workers, pdf_writer)
        
        if no_client_training:
            break

    pdf_writer.close()
    epech_stop=epoch
    return epech_stop
    
