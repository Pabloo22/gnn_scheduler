create_image:
	@echo "Creating image..."
	@docker build -t alpha_graph_image .

run_container:
	@echo "Running container..."
	@docker run -d --name alpha_graph_container -v $(PWD):/path/in/container alpha_graph_image

start_container:
	@echo "Starting container..."
	@docker start alpha_graph_container
