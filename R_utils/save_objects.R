load_objects <- function(example, file)
{
    examp_dir <- here::here("examples", example, "R")
    load(file.path(examp_dir, file), envir=globalenv())
}

save_objects <- function(..., example, file)
{
    examp_dir <- here::here("examples", example, "R")
    save(..., file=file.path(examp_dir, file))
}
