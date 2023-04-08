from copy import copy
from typing import Callable, List, Tuple, Optional

import click

from dbt.cli import requires, params as p
from dbt.config.profile import Profile
from dbt.config.project import Project
from dbt.contracts.graph.manifest import Manifest
from dbt.events.base_types import EventMsg
from dbt.task.build import BuildTask
from dbt.task.clean import CleanTask
from dbt.task.compile import CompileTask
from dbt.task.debug import DebugTask
from dbt.task.deps import DepsTask
from dbt.task.freshness import FreshnessTask
from dbt.task.generate import GenerateTask
from dbt.task.init import InitTask
from dbt.task.list import ListTask
from dbt.task.run import RunTask
from dbt.task.run_operation import RunOperationTask
from dbt.task.seed import SeedTask
from dbt.task.serve import ServeTask
from dbt.task.show import ShowTask
from dbt.task.snapshot import SnapshotTask
from dbt.task.test import TestTask


class dbtUsageException(Exception):
    pass


class dbtInternalException(Exception):
    pass


# Programmatic invocation
class dbtRunner:
    def __init__(
        self,
        project: Project = None,
        profile: Profile = None,
        manifest: Manifest = None,
        callbacks: List[Callable[[EventMsg], None]] = [],
    ):
        self.project = project
        self.profile = profile
        self.manifest = manifest
        self.callbacks = callbacks

    def invoke(self, args: List[str], **kwargs) -> Tuple[Optional[List], bool]:
        try:
            dbt_ctx = cli.make_context(cli.name, args)
            dbt_ctx.obj = {
                "project": self.project,
                "profile": self.profile,
                "manifest": self.manifest,
                "callbacks": self.callbacks,
            }

            for key, value in kwargs.items():
                dbt_ctx.params[key] = value
                # Hack to set parameter source to custom string
                dbt_ctx.set_parameter_source(key, "kwargs")  # type: ignore

            return cli.invoke(dbt_ctx)
        except requires.HandledExit as e:
            return (e.result, e.success)
        except requires.UnhandledExit as e:
            raise e.exception
        except click.exceptions.Exit as e:
            # 0 exit code, expected for --version early exit
            if str(e) == "0":
                return [], True
            raise dbtInternalException(f"unhandled exit code {str(e)}")
        except (click.NoSuchOption, click.UsageError) as e:
            raise dbtUsageException(e.message)


# dbt
@click.group(
    context_settings={"help_option_names": ["-h", "--help"]},
    invoke_without_command=True,
    no_args_is_help=True,
    epilog="Specify one of these sub-commands and you can find more help from there.",
)
@click.pass_context
@p.send_anonymous_usage_stats
@p.cache_selected_only
@p.debug
@p.enable_legacy_logger
@p.fail_fast
@p.log_cache_events
@p.log_format
@p.log_format_file
@p.log_level
@p.log_level_file
@p.log_path
@p.macro_debugging
@p.partial_parse
@p.print
@p.deprecated_print
@p.printer_width
@p.quiet
@p.record_timing_info
@p.single_threaded
@p.static_parser
@p.use_colors
@p.use_colors_file
@p.use_experimental_parser
@p.version
@p.version_check
@p.warn_error
@p.warn_error_options
@p.write_json
def cli(ctx, **kwargs):
    """An ELT tool for managing your SQL transformations and data models.
    For more documentation on these commands, visit: docs.getdbt.com
    """


# dbt build
@cli.command("build")
@click.pass_context
@p.defer
@p.deprecated_defer
@p.exclude
@p.fail_fast
@p.favor_state
@p.deprecated_favor_state
@p.full_refresh
@p.indirect_selection
@p.profile
@p.profiles_dir
@p.project_dir
@p.resource_type
@p.select
@p.selector
@p.show
@p.state
@p.deprecated_state
@p.store_failures
@p.target
@p.target_path
@p.threads
@p.vars
@p.version_check
@requires.preflight
@requires.profile
@requires.project
@requires.runtime_config
@requires.manifest
@requires.postflight
def build(ctx, **kwargs):
    """Run all Seeds, Models, Snapshots, and tests in DAG order"""
    task = BuildTask(
        ctx.obj["flags"],
        ctx.obj["runtime_config"],
        ctx.obj["manifest"],
    )

    results = task.run()
    success = task.interpret_results(results)
    return results, success


# dbt clean
@cli.command("clean")
@click.pass_context
@p.profile
@p.profiles_dir
@p.project_dir
@p.target
@p.vars
@requires.preflight
@requires.unset_profile
@requires.project
@requires.postflight
def clean(ctx, **kwargs):
    """Delete all folders in the clean-targets list (usually the dbt_packages and target directories.)"""
    task = CleanTask(ctx.obj["flags"], ctx.obj["project"])

    results = task.run()
    success = task.interpret_results(results)
    return results, success


# dbt docs
@cli.group()
@click.pass_context
def docs(ctx, **kwargs):
    """Generate or serve the documentation website for your project"""


# dbt docs generate
@docs.command("generate")
@click.pass_context
@p.compile_docs
@p.defer
@p.deprecated_defer
@p.exclude
@p.favor_state
@p.deprecated_favor_state
@p.profile
@p.profiles_dir
@p.project_dir
@p.select
@p.selector
@p.state
@p.deprecated_state
@p.target
@p.target_path
@p.threads
@p.vars
@p.version_check
@requires.preflight
@requires.profile
@requires.project
@requires.runtime_config
@requires.manifest(write=False)
@requires.postflight
def docs_generate(ctx, **kwargs):
    """Generate the documentation website for your project"""
    task = GenerateTask(
        ctx.obj["flags"],
        ctx.obj["runtime_config"],
        ctx.obj["manifest"],
    )

    results = task.run()
    success = task.interpret_results(results)
    return results, success


# dbt docs serve
@docs.command("serve")
@click.pass_context
@p.browser
@p.port
@p.profile
@p.profiles_dir
@p.project_dir
@p.target
@p.vars
@requires.preflight
@requires.profile
@requires.project
@requires.runtime_config
@requires.manifest
@requires.postflight
def docs_serve(ctx, **kwargs):
    """Serve the documentation website for your project"""
    task = ServeTask(
        ctx.obj["flags"],
        ctx.obj["runtime_config"],
        ctx.obj["manifest"],
    )

    results = task.run()
    success = task.interpret_results(results)
    return results, success


# dbt compile
@cli.command("compile")
@click.pass_context
@p.defer
@p.deprecated_defer
@p.exclude
@p.favor_state
@p.deprecated_favor_state
@p.full_refresh
@p.show_output_format
@p.indirect_selection
@p.introspect
@p.parse_only
@p.profile
@p.profiles_dir
@p.project_dir
@p.select
@p.selector
@p.inline
@p.state
@p.deprecated_state
@p.target
@p.target_path
@p.threads
@p.vars
@p.version_check
@requires.preflight
@requires.profile
@requires.project
@requires.runtime_config
@requires.manifest
@requires.postflight
def compile(ctx, **kwargs):
    """Generates executable SQL from source, model, test, and analysis files. Compiled SQL files are written to the
    target/ directory."""
    task = CompileTask(
        ctx.obj["flags"],
        ctx.obj["runtime_config"],
        ctx.obj["manifest"],
    )

    results = task.run()
    success = task.interpret_results(results)
    return results, success


# dbt show
@cli.command("show")
@click.pass_context
@p.defer
@p.deprecated_defer
@p.exclude
@p.favor_state
@p.deprecated_favor_state
@p.full_refresh
@p.show_output_format
@p.show_limit
@p.indirect_selection
@p.introspect
@p.parse_only
@p.profile
@p.profiles_dir
@p.project_dir
@p.select
@p.selector
@p.inline
@p.state
@p.deprecated_state
@p.target
@p.target_path
@p.threads
@p.vars
@p.version_check
@requires.preflight
@requires.profile
@requires.project
@requires.runtime_config
@requires.manifest
@requires.postflight
def show(ctx, **kwargs):
    """Generates executable SQL for a named resource or inline query, runs that SQL, and returns a preview of the
    results. Does not materialize anything to the warehouse."""
    task = ShowTask(
        ctx.obj["flags"],
        ctx.obj["runtime_config"],
        ctx.obj["manifest"],
    )

    results = task.run()
    success = task.interpret_results(results)
    return results, success


# dbt debug
@cli.command("debug")
@click.pass_context
@p.config_dir
@p.profile
@p.profiles_dir_exists_false
@p.project_dir
@p.target
@p.vars
@p.version_check
@requires.preflight
@requires.postflight
def debug(ctx, **kwargs):
    """Show some helpful information about dbt for debugging. Not to be confused with the --debug option which increases verbosity."""
    task = DebugTask(
        ctx.obj["flags"],
        None,
    )

    results = task.run()
    success = task.interpret_results(results)
    return results, success


# dbt deps
@cli.command("deps")
@click.pass_context
@p.profile
@p.profiles_dir
@p.project_dir
@p.target
@p.vars
@requires.preflight
@requires.unset_profile
@requires.project
@requires.postflight
def deps(ctx, **kwargs):
    """Pull the most recent version of the dependencies listed in packages.yml"""
    task = DepsTask(ctx.obj["flags"], ctx.obj["project"])
    results = task.run()
    success = task.interpret_results(results)
    return results, success


# dbt init
@cli.command("init")
@click.pass_context
# for backwards compatibility, accept 'project_name' as an optional positional argument
@click.argument("project_name", required=False)
@p.profile
@p.profiles_dir
@p.project_dir
@p.skip_profile_setup
@p.target
@p.vars
@requires.preflight
@requires.postflight
def init(ctx, **kwargs):
    """Initialize a new dbt project."""
    task = InitTask(ctx.obj["flags"], None)

    results = task.run()
    success = task.interpret_results(results)
    return results, success


# dbt list
@cli.command("list")
@click.pass_context
@p.exclude
@p.indirect_selection
@p.models
@p.output
@p.output_keys
@p.profile
@p.profiles_dir
@p.project_dir
@p.resource_type
@p.raw_select
@p.selector
@p.state
@p.deprecated_state
@p.target
@p.vars
@requires.preflight
@requires.profile
@requires.project
@requires.runtime_config
@requires.manifest
@requires.postflight
def list(ctx, **kwargs):
    """List the resources in your project"""
    task = ListTask(
        ctx.obj["flags"],
        ctx.obj["runtime_config"],
        ctx.obj["manifest"],
    )

    results = task.run()
    success = task.interpret_results(results)
    return results, success


# Alias "list" to "ls"
ls = copy(cli.commands["list"])
ls.hidden = True
cli.add_command(ls, "ls")


# dbt parse
@cli.command("parse")
@click.pass_context
@p.compile_parse
@p.profile
@p.profiles_dir
@p.project_dir
@p.target
@p.target_path
@p.threads
@p.vars
@p.version_check
@p.write_manifest
@requires.preflight
@requires.profile
@requires.project
@requires.runtime_config
@requires.manifest(write_perf_info=True)
@requires.postflight
def parse(ctx, **kwargs):
    """Parses the project and provides information on performance"""
    # manifest generation and writing happens in @requires.manifest
    return None, True


# dbt run
@cli.command("run")
@click.pass_context
@p.defer
@p.deprecated_defer
@p.favor_state
@p.deprecated_favor_state
@p.exclude
@p.fail_fast
@p.full_refresh
@p.profile
@p.profiles_dir
@p.project_dir
@p.select
@p.selector
@p.state
@p.deprecated_state
@p.target
@p.target_path
@p.threads
@p.vars
@p.version_check
@requires.preflight
@requires.profile
@requires.project
@requires.runtime_config
@requires.manifest
@requires.postflight
def run(ctx, **kwargs):
    """Compile SQL and execute against the current target database."""
    task = RunTask(
        ctx.obj["flags"],
        ctx.obj["runtime_config"],
        ctx.obj["manifest"],
    )

    results = task.run()
    success = task.interpret_results(results)
    return results, success


# dbt run operation
@cli.command("run-operation")
@click.pass_context
@click.argument("macro")
@p.args
@p.profile
@p.profiles_dir
@p.project_dir
@p.target
@p.vars
@requires.preflight
@requires.profile
@requires.project
@requires.runtime_config
@requires.manifest
@requires.postflight
def run_operation(ctx, **kwargs):
    """Run the named macro with any supplied arguments."""
    task = RunOperationTask(
        ctx.obj["flags"],
        ctx.obj["runtime_config"],
        ctx.obj["manifest"],
    )

    results = task.run()
    success = task.interpret_results(results)
    return results, success


# dbt seed
@cli.command("seed")
@click.pass_context
@p.exclude
@p.full_refresh
@p.profile
@p.profiles_dir
@p.project_dir
@p.select
@p.selector
@p.show
@p.state
@p.deprecated_state
@p.target
@p.target_path
@p.threads
@p.vars
@p.version_check
@requires.preflight
@requires.profile
@requires.project
@requires.runtime_config
@requires.manifest
@requires.postflight
def seed(ctx, **kwargs):
    """Load data from csv files into your data warehouse."""
    task = SeedTask(
        ctx.obj["flags"],
        ctx.obj["runtime_config"],
        ctx.obj["manifest"],
    )
    results = task.run()
    success = task.interpret_results(results)
    return results, success


# dbt snapshot
@cli.command("snapshot")
@click.pass_context
@p.defer
@p.deprecated_defer
@p.exclude
@p.favor_state
@p.deprecated_favor_state
@p.profile
@p.profiles_dir
@p.project_dir
@p.select
@p.selector
@p.state
@p.deprecated_state
@p.target
@p.threads
@p.vars
@requires.preflight
@requires.profile
@requires.project
@requires.runtime_config
@requires.manifest
@requires.postflight
def snapshot(ctx, **kwargs):
    """Execute snapshots defined in your project"""
    task = SnapshotTask(
        ctx.obj["flags"],
        ctx.obj["runtime_config"],
        ctx.obj["manifest"],
    )

    results = task.run()
    success = task.interpret_results(results)
    return results, success


# dbt source
@cli.group()
@click.pass_context
def source(ctx, **kwargs):
    """Manage your project's sources"""


# dbt source freshness
@source.command("freshness")
@click.pass_context
@p.exclude
@p.output_path  # TODO: Is this ok to re-use?  We have three different output params, how much can we consolidate?
@p.profile
@p.profiles_dir
@p.project_dir
@p.select
@p.selector
@p.state
@p.deprecated_state
@p.target
@p.threads
@p.vars
@requires.preflight
@requires.profile
@requires.project
@requires.runtime_config
@requires.manifest
@requires.postflight
def freshness(ctx, **kwargs):
    """check the current freshness of the project's sources"""
    task = FreshnessTask(
        ctx.obj["flags"],
        ctx.obj["runtime_config"],
        ctx.obj["manifest"],
    )

    results = task.run()
    success = task.interpret_results(results)
    return results, success


# Alias "source freshness" to "snapshot-freshness"
snapshot_freshness = copy(cli.commands["source"].commands["freshness"])  # type: ignore
snapshot_freshness.hidden = True
cli.commands["source"].add_command(snapshot_freshness, "snapshot-freshness")  # type: ignore


# dbt test
@cli.command("test")
@click.pass_context
@p.defer
@p.deprecated_defer
@p.exclude
@p.fail_fast
@p.favor_state
@p.deprecated_favor_state
@p.indirect_selection
@p.profile
@p.profiles_dir
@p.project_dir
@p.select
@p.selector
@p.state
@p.deprecated_state
@p.store_failures
@p.target
@p.target_path
@p.threads
@p.vars
@p.version_check
@requires.preflight
@requires.profile
@requires.project
@requires.runtime_config
@requires.manifest
@requires.postflight
def test(ctx, **kwargs):
    """Runs tests on data in deployed models. Run this after `dbt run`"""
    task = TestTask(
        ctx.obj["flags"],
        ctx.obj["runtime_config"],
        ctx.obj["manifest"],
    )

    results = task.run()
    success = task.interpret_results(results)
    return results, success


# Support running as a module
if __name__ == "__main__":
    cli()
